#!/usr/bin/env python3
"""Aggregate RiverEdge restore, reproduction, and profiler artifacts."""

from __future__ import annotations

import csv
import json
import re
import sqlite3
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
HISTORICAL_ROOT = ROOT.parent / "11_vllm_batch_cudagraph"

NSYS_METRICS = {
    "GPC Clock Frequency [MHz]": "gpc_clock_mhz",
    "GPU Active [Throughput %]": "gpu_active_pct",
    "GR Active [Throughput %]": "gr_active_pct",
    "SMs Active [Throughput %]": "sms_active_pct",
    "SM Issue [Throughput %]": "sm_issue_pct",
    "Tensor Active [Throughput %]": "tensor_active_pct",
    "Compute Warps in Flight [Throughput %]": "compute_warps_pct",
    "Unallocated Warps in Active SMs [Throughput %]": "unallocated_warps_pct",
}

NCU_METRICS = {
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": (
        "compute_memory_throughput_pct"
    ),
    "gpu__time_duration.sum": "duration_ns",
    "lts__d_sectors_fill_device.sum": "device_fill_sectors",
    "lts__t_bytes.sum": "l2_bytes",
    "lts__t_bytes.sum.per_second": "l2_bytes_per_s",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_throughput_pct",
    "smsp__issue_active.avg.pct_of_peak_sustained_active": "sm_issue_pct",
    "smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed": (
        "tensor_cycles_pct"
    ),
    "smsp__warps_active.avg.pct_of_peak_sustained_active": "warps_active_pct",
}

TEGRASTATS_PATTERN = re.compile(
    r"EMC_FREQ (\d+)%@(\d+) GR3D_FREQ (\d+)%@\[(\d+),(\d+)\]"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reproduction_comparison() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    historical = []
    for execution_mode in ("eager", "cudagraph"):
        historical.extend(read_csv(HISTORICAL_ROOT / f"{execution_mode}_matrix.csv"))
    reproduced = read_csv(ROOT / "reproduction_matrix.csv")

    def key(row: dict[str, str]) -> tuple[str, str, int]:
        return row["mode"], row["execution_mode"], int(row["batch_size"])

    old_by_key = {key(row): row for row in historical}
    new_by_key = {key(row): row for row in reproduced}
    comparisons = []
    for row_key in sorted(new_by_key, key=lambda value: (value[1], value[0], value[2])):
        old_tps = float(old_by_key[row_key]["output_tps"])
        new_tps = float(new_by_key[row_key]["output_tps"])
        comparisons.append(
            {
                "execution_mode": row_key[1],
                "mode": row_key[0],
                "batch_size": row_key[2],
                "historical_tps": round(old_tps, 6),
                "reproduced_tps": round(new_tps, 6),
                "delta_pct": round((new_tps / old_tps - 1.0) * 100.0, 4),
            }
        )

    speedups = []
    for execution_mode in ("eager", "cudagraph"):
        for batch_size in (1, 2, 4, 8, 16):
            fp_tps = float(new_by_key[("full_fp", execution_mode, batch_size)]["output_tps"])
            ptq_tps = float(
                new_by_key[("static_ptq_tail", execution_mode, batch_size)]["output_tps"]
            )
            speedups.append(
                {
                    "execution_mode": execution_mode,
                    "batch_size": batch_size,
                    "full_fp_tps": round(fp_tps, 6),
                    "static_ptq_tail_tps": round(ptq_tps, 6),
                    "ptq_over_fp": round(ptq_tps / fp_tps, 4),
                }
            )
    return comparisons, speedups


def nsys_metrics() -> list[dict[str, object]]:
    rows = []
    for mode in ("full_fp", "static_ptq_tail"):
        database = ROOT / "nsys" / f"{mode}_b1_b2_b4.sqlite"
        with sqlite3.connect(database) as connection:
            metric_ids = {
                name: metric_id
                for metric_id, name in connection.execute(
                    "SELECT metricId, metricName FROM TARGET_INFO_GPU_METRICS"
                )
            }
            for batch_size in (1, 2, 4):
                range_name = f"riveredge_{mode}_cudagraph_batch{batch_size}"
                range_row = connection.execute(
                    "SELECT start, end FROM NVTX_EVENTS WHERE text = ?", (range_name,)
                ).fetchone()
                if range_row is None:
                    raise RuntimeError(f"missing NVTX range {range_name} in {database}")
                start, end = range_row
                row: dict[str, object] = {
                    "mode": mode,
                    "batch_size": batch_size,
                }
                for metric_name, output_name in NSYS_METRICS.items():
                    metric_id = metric_ids[metric_name]
                    average, samples = connection.execute(
                        """
                        SELECT AVG(value), COUNT(*)
                        FROM GPU_METRICS
                        WHERE metricId = ? AND timestamp BETWEEN ? AND ?
                        """,
                        (metric_id, start, end),
                    ).fetchone()
                    if metric_name.startswith("GPC Clock"):
                        average /= 1_000_000.0
                    row[output_name] = round(average, 4)
                    row["samples"] = samples
                rows.append(row)
    return rows


def nsys_memcpy_bytes() -> list[dict[str, object]]:
    rows = []
    copy_names = {1: "host_to_device", 2: "device_to_host", 8: "device_to_device"}
    for mode in ("full_fp", "static_ptq_tail"):
        database = ROOT / "nsys" / f"{mode}_eager_b1_kernel_id.sqlite"
        with sqlite3.connect(database) as connection:
            range_name = f"riveredge_{mode}_eager_batch1"
            start, end = connection.execute(
                "SELECT start, end FROM NVTX_EVENTS WHERE text = ?", (range_name,)
            ).fetchone()
            for copy_kind, count, total_bytes, max_bytes in connection.execute(
                """
                SELECT copyKind, COUNT(*), SUM(bytes), MAX(bytes)
                FROM CUPTI_ACTIVITY_KIND_MEMCPY
                WHERE start >= ? AND end <= ?
                GROUP BY copyKind
                """,
                (start, end),
            ):
                rows.append(
                    {
                        "mode": mode,
                        "operation": copy_names.get(copy_kind, f"copy_kind_{copy_kind}"),
                        "count": count,
                        "total_bytes": total_bytes,
                        "max_bytes": max_bytes,
                    }
                )
    return rows


def ncu_metrics() -> list[dict[str, object]]:
    rows = []
    for mode, filename in (
        ("full_fp", "fp_gate_up_b1_metrics.csv"),
        ("static_ptq_tail", "ptq_gate_up_b1_metrics.csv"),
    ):
        source_rows = read_csv(ROOT / "ncu" / filename)
        values = {
            NCU_METRICS[row["Metric Name"]]: float(row["Metric Value"])
            for row in source_rows
            if row["Metric Name"] in NCU_METRICS
        }
        first = source_rows[0]
        row: dict[str, object] = {
            "mode": mode,
            "kernel": first["Kernel Name"],
            "grid_size": first["Grid Size"],
            "block_size": first["Block Size"],
        }
        row.update({name: round(value, 6) for name, value in values.items()})
        row["duration_ms"] = round(values["duration_ns"] / 1_000_000.0, 6)
        row["l2_traffic_mb"] = round(values["l2_bytes"] / 1_000_000.0, 6)
        row["l2_throughput_gb_s"] = round(values["l2_bytes_per_s"] / 1e9, 6)
        rows.append(row)
    return rows


def tegrastats_metrics() -> list[dict[str, object]]:
    rows = []
    for mode in ("full_fp", "static_ptq_tail"):
        for batch_size in (1, 4):
            stem = f"{mode}_b{batch_size}"
            samples = []
            for line in (ROOT / "tegrastats" / f"{stem}.log").read_text().splitlines():
                match = TEGRASTATS_PATTERN.search(line)
                if match:
                    samples.append(tuple(map(int, match.groups())))

            active_runs: list[list[tuple[int, ...]]] = []
            active_run: list[tuple[int, ...]] = []
            for sample in samples:
                if sample[2] >= 90:
                    active_run.append(sample)
                elif active_run:
                    active_runs.append(active_run)
                    active_run = []
            if active_run:
                active_runs.append(active_run)
            if not active_runs:
                raise RuntimeError(f"no sustained GPU-active samples in {stem}.log")

            longest_run = max(active_runs, key=len)
            trim_samples = min(10, max(0, (len(longest_run) - 20) // 2))
            steady = (
                longest_run[trim_samples:-trim_samples]
                if trim_samples
                else longest_run
            )
            emc = sorted(sample[0] for sample in steady)
            gr3d = [sample[2] for sample in steady]
            run_data = json.loads(
                (ROOT / "tegrastats" / f"{stem}_run.json").read_text(encoding="utf-8")
            )
            rows.append(
                {
                    "mode": mode,
                    "batch_size": batch_size,
                    "output_tps": round(run_data["rows"][0]["output_tps"], 6),
                    "sample_interval_ms": 100,
                    "steady_samples": len(steady),
                    "trim_samples_each_side": trim_samples,
                    "emc_util_mean_pct": round(statistics.mean(emc), 4),
                    "emc_util_median_pct": round(statistics.median(emc), 4),
                    "emc_util_p5_pct": emc[int(0.05 * (len(emc) - 1))],
                    "emc_util_p95_pct": emc[int(0.95 * (len(emc) - 1))],
                    "emc_clock_mhz": round(
                        statistics.median(sample[1] for sample in steady), 4
                    ),
                    "gr3d_util_mean_pct": round(statistics.mean(gr3d), 4),
                }
            )
    return rows


def main() -> None:
    comparisons, speedups = reproduction_comparison()
    nsys_rows = nsys_metrics()
    memcpy_rows = nsys_memcpy_bytes()
    ncu_rows = ncu_metrics()
    tegrastats_rows = tegrastats_metrics()

    write_csv(ROOT / "reproduction_comparison.csv", comparisons)
    write_csv(ROOT / "reproduction_speedups.csv", speedups)
    write_csv(ROOT / "nsys_gpu_metrics.csv", nsys_rows)
    write_csv(ROOT / "nsys_memcpy_bytes.csv", memcpy_rows)
    write_csv(ROOT / "ncu_kernel_metrics.csv", ncu_rows)
    write_csv(ROOT / "tegrastats_emc_metrics.csv", tegrastats_rows)

    ncu_by_mode = {row["mode"]: row for row in ncu_rows}
    summary = {
        "max_abs_reproduction_delta_pct": max(
            abs(float(row["delta_pct"])) for row in comparisons
        ),
        "reproduction_points": len(comparisons),
        "ncu_gate_up_l2_traffic_reduction": round(
            float(ncu_by_mode["full_fp"]["l2_bytes"])
            / float(ncu_by_mode["static_ptq_tail"]["l2_bytes"]),
            4,
        ),
        "comparisons": comparisons,
        "speedups": speedups,
        "nsys_gpu_metrics": nsys_rows,
        "nsys_memcpy_bytes": memcpy_rows,
        "ncu_kernel_metrics": ncu_rows,
        "tegrastats_emc_metrics": tegrastats_rows,
    }
    (ROOT / "validation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
