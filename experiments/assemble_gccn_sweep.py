#!/usr/bin/env python
"""Fold the MANTRA GCCN sweep into data/frozen/baseline_markers.parquet.

Reads results/gccn_sweep_mantra/sweep_*.jsonl (23 configs x 3 seeds,
run_gccn_sweep.py) and appends gccn_sweep_best rows in the frame's
per-seed convention: per seed, the config with the best plain
validation accuracy wins; its plain and balanced test accuracies are
recorded; cost is the whole grid priced once (sum over configs of the
per-config mean measured FLOPs / wall seconds, flops_estimated=False).
Idempotent: existing (mantra_orientation, gccn_sweep_best) rows are
replaced.

    python experiments/assemble_gccn_sweep.py [--allow-missing]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
FRAME = REPO / "data" / "frozen" / "baseline_markers.parquet"
SEEDS = [42, 43, 44]
N_CONFIGS = 23

# dataset -> results dir. MUTAG/NCI109/alkane added 2026-09-23: real
# sweeps replace the enumeration stand-ins in the lead figure/table.
DATASETS = {
    "mantra_orientation": REPO / "results" / "gccn_sweep_mantra",
    "MUTAG": REPO / "results" / "gccn_sweep_t1",
    "NCI109": REPO / "results" / "gccn_sweep_t1",
    "alkane_carbonyl": REPO / "results" / "gccn_sweep_t1",
}


def dataset_rows(ds: str, results: Path, allow_missing: bool) -> list[dict]:
    rows, missing = [], []
    for i in range(N_CONFIGS):
        path = results / f"sweep_{ds}_cfg{i:02d}.jsonl"
        if not path.exists():
            missing.append(f"cfg{i:02d}")
            continue
        for line in path.read_text().splitlines():
            rows.append(json.loads(line))
    if missing:
        print(f"{ds}: missing {len(missing)} configs: {missing}",
              file=sys.stderr)
        if not allow_missing:
            sys.exit(2)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-missing", action="store_true")
    args = ap.parse_args()

    for DS, RESULTS in DATASETS.items():
        rows = dataset_rows(DS, RESULTS, args.allow_missing)
        if not rows:
            print(f"{DS}: no rows, skipped", file=sys.stderr)
            continue
        fold_dataset(DS, rows)


def fold_dataset(DS: str, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    df["total_flops"] = df.flops.map(lambda f: float(f["total_flops"]))

    per_cfg = df.groupby("config_tag").agg(
        flops=("total_flops", "mean"), wall=("train_seconds", "mean"))
    grid_flops = float(per_cfg.flops.sum())
    grid_wall = float(per_cfg.wall.sum())

    out = []
    for seed, g in df.groupby("seed"):
        if len(g) < N_CONFIGS:
            print(f"seed {seed}: only {len(g)}/{N_CONFIGS} configs",
                  file=sys.stderr)
        best = g.loc[g.val_accuracy.idxmax()]
        out.append(dict(
            dataset=DS, baseline="gccn_sweep_best",
            n_configs=len(g), flops=grid_flops, wall_seconds=grid_wall,
            flops_estimated=False, seed=int(seed),
            test_accuracy=float(best.test_accuracy),
            val_accuracy=float(best.val_accuracy),
            test_balanced_accuracy=float(best.test_balanced_accuracy),
            winner=str(best.coalition),
        ))

    bm = pd.read_parquet(FRAME)
    bm = bm[~((bm.dataset == DS) & (bm.baseline == "gccn_sweep_best"))]
    bm = pd.concat([bm, pd.DataFrame(out)], ignore_index=True)
    bm.to_parquet(FRAME, index=False)
    print(f"appended {len(out)} {DS} gccn_sweep_best rows to {FRAME} "
          f"(grid flops {grid_flops:.3e}, {len(df)} runs)")


if __name__ == "__main__":
    main()
