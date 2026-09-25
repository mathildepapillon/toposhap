#!/usr/bin/env python
"""Compute-matched shortlist baselines for the selection procedure (v3 review).

    python experiments/run_baseline_shortlists.py --dataset benzene --seed 42

Two arms, sharing ONE 5-epoch stem per (dataset, seed), so their compute
matches the ladder B=2 arm's stem + continuations:

* random_b2    two candidate architectures drawn uniformly at random from
               the 2047 non-empty neighborhood subsets (no masked-game
               walk at all); both warm-pruned from the stem, continued to
               the standard budget, winner by validation accuracy.
* full_default the prespecified one-run default: all 11 neighborhoods,
               continued from the same stem (B=1, no search).

RNG is seeded by (dataset, seed) so the draw is reproducible. Balanced
metrics are recorded in-run for every dataset (needed on MANTRA).
Outputs are kept OUT of the frozen frames: results/baseline_shortlists/.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

os.environ.setdefault("TOPOSHAP_INTERRANK_ORIENTATION", "fixed")
os.environ.setdefault(
    "TOPOSHAP_GRAPHXAI_DATA",
    str(REPO / "external" / "TopoBench" / "datasets" / "graphxai"))

from toposhap.patches import apply_all  # noqa: E402

PATCHES = apply_all()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import topobench_driver as drv  # noqa: E402
from mantra_balanced_eval import split_metrics  # noqa: E402
from resume_guard import cell_complete  # noqa: E402
from run_autok import DATASETS  # noqa: E402
from run_ladder_b2 import continue_and_test  # noqa: E402
from toposhap.vocabulary import FULL_MASK, N_PLAYERS, NEIGHBORHOODS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--stem-epochs", type=int, default=5)
    ap.add_argument("--model", default=drv.RECORD_MODEL.split("/", 1)[1])
    ap.add_argument("--out-dir",
                    default=str(REPO / "results" / "baseline_shortlists"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shortlists_{args.dataset}_seed{args.seed}.jsonl"
    if cell_complete(out_path):
        print(f"skip: {out_path} already complete")
        return
    run_dir = out_dir / "runs" / f"{args.dataset}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"patches: {PATCHES}")

    # ---- shared stem (identical settings to run_ladder_b2) ----------------
    model_name = f"{drv.model_domain(args.dataset)}/{args.model}"
    stem_cfg = drv.compose_config(
        dataset=args.dataset, seed=args.seed,
        neighborhoods=list(NEIGHBORHOODS),
        output_dir=str(run_dir / "stem"), model=model_name)
    pipe = drv.build_pipeline(stem_cfg)
    stem_seconds = drv.fit_stem(stem_cfg, pipe, epochs=args.stem_epochs)
    stem = pipe.model.to(device)

    # ---- reproducible random draw: two distinct non-empty masks -----------
    draw_seed = zlib.adler32(f"{args.dataset}|{args.seed}".encode())
    rng = np.random.default_rng(draw_seed)
    masks = []
    while len(masks) < 2:
        m = int(rng.integers(1, FULL_MASK + 1))
        if m not in masks:
            masks.append(m)

    def run_candidate(tag: str, mask: int) -> dict:
        model = copy.deepcopy(stem)
        t0 = time.perf_counter()
        res = continue_and_test(
            args.dataset, args.seed, model, pipe.datamodule, mask,
            run_dir / f"continue_{tag}", device, model_name=model_name)
        res["mask"] = mask
        res["k"] = bin(mask).count("1")
        res["tag"] = tag
        # balanced metrics from the tested best checkpoint (pipe.model
        # holds it after continue_and_test -> test_best_checkpoint)
        vm = split_metrics(model, pipe.datamodule.val_dataloader(),
                           "Validation", device)
        tm = split_metrics(model, pipe.datamodule.test_dataloader(),
                           "Test", device)
        res["val_balanced_accuracy"] = float(vm["balanced"])
        res["test_balanced_accuracy"] = float(tm["balanced"])
        res["wall_seconds_candidate"] = time.perf_counter() - t0
        return res

    rows = []
    # arm 1: random B=2 shortlist
    c1 = run_candidate("rand1", masks[0])
    c2 = run_candidate("rand2", masks[1])
    winner = c1 if c1["val"] >= c2["val"] else c2
    rows.append({
        "dataset": args.dataset, "seed": args.seed,
        "selector": "random_b2", "draw_seed": int(draw_seed),
        "candidates": [
            {k: c[k] for k in ("tag", "mask", "k", "val", "test",
                               "val_balanced_accuracy",
                               "test_balanced_accuracy")}
            for c in (c1, c2)],
        "mask": winner["mask"], "k": winner["k"],
        "val_accuracy": float(winner["val"]),
        "test_accuracy": float(winner["test"]),
        "val_balanced_accuracy": winner["val_balanced_accuracy"],
        "test_balanced_accuracy": winner["test_balanced_accuracy"],
        "stem_seconds": float(stem_seconds),
        "stem_epochs": int(args.stem_epochs),
        "model": args.model, "patches": PATCHES,
        "note": "compute-matched to ladder B=2 minus the masked-game walk",
    })
    # arm 2: prespecified full-vocabulary default (one run)
    cf = run_candidate("full", FULL_MASK)
    rows.append({
        "dataset": args.dataset, "seed": args.seed,
        "selector": "full_default",
        "mask": FULL_MASK, "k": N_PLAYERS,
        "val_accuracy": float(cf["val"]),
        "test_accuracy": float(cf["test"]),
        "val_balanced_accuracy": cf["val_balanced_accuracy"],
        "test_balanced_accuracy": cf["test_balanced_accuracy"],
        "stem_seconds": float(stem_seconds),
        "stem_epochs": int(args.stem_epochs),
        "model": args.model, "patches": PATCHES,
        "note": "prespecified default: all neighborhoods, stem + one "
                "continuation, no search",
    })
    with open(out_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {out_path}")
    for r in rows:
        print(f"  {r['selector']}: k={r['k']} mask={r['mask']} "
              f"val={r['val_accuracy']:.4f} test={r['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
