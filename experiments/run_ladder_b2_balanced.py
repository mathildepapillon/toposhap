#!/usr/bin/env python
"""MANTRA supplementary arm: ladder B=2 with BALANCED-accuracy selection.

    python experiments/run_ladder_b2_balanced.py --seed 42

Identical to run_ladder_b2.py on mantra_orientation except that every
selection-relevant score is balanced validation accuracy (mean per-class
recall): the masked game, the stop threshold's p, and the final
comparison of trained candidates. Answers the metric-mismatch question
(review #17): does selecting on the reported metric remove the
majority-collapse weakness? The frozen plain-accuracy run is untouched;
outputs land in results/ladder_b2_mantra_balanced/ with selector
"ladder_b2_balanced". The binomial floor with balanced p is an
approximation (balanced accuracy is not a binomial proportion) and is
labeled as such wherever reported.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

os.environ.setdefault("TOPOSHAP_INTERRANK_ORIENTATION", "fixed")

from toposhap.patches import apply_all  # noqa: E402

PATCHES = apply_all()

import torch  # noqa: E402

import topobench_driver as drv  # noqa: E402
from mantra_balanced_eval import split_metrics  # noqa: E402
from resume_guard import cell_complete  # noqa: E402
from run_ladder_b2 import backward_ladder, continue_and_test  # noqa: E402
from toposhap.neighborhoods import CoalitionMaskedBackbone  # noqa: E402
from toposhap.shapley.games import CachedGame  # noqa: E402
from toposhap.vocabulary import FULL_MASK, N_PLAYERS, NEIGHBORHOODS  # noqa: E402

DS = "mantra_orientation"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--stem-epochs", type=int, default=5)
    ap.add_argument("--model", default=drv.RECORD_MODEL.split("/", 1)[1])
    ap.add_argument("--out-dir",
                    default=str(REPO / "results" / "ladder_b2_mantra_balanced"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ladder_b2_balanced_{DS}_seed{args.seed}.jsonl"
    if cell_complete(out_path):
        print(f"skip: {out_path} already complete")
        return
    run_dir = out_dir / "runs" / f"{DS}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"patches: {PATCHES}")

    model_name = f"{drv.model_domain(DS)}/{args.model}"
    stem_cfg = drv.compose_config(
        dataset=DS, seed=args.seed, neighborhoods=list(NEIGHBORHOODS),
        output_dir=str(run_dir / "stem"), model=model_name)
    pipe = drv.build_pipeline(stem_cfg)
    stem_seconds = drv.fit_stem(stem_cfg, pipe, epochs=args.stem_epochs)
    model = pipe.model.to(device)

    inner = model.backbone.backbone
    wrapper = CoalitionMaskedBackbone(inner)
    val_loader = pipe.datamodule.val_dataloader()
    m_val = {}

    def evaluate(mask: int) -> float:
        with wrapper.coalition(mask):
            m = split_metrics(model, val_loader, "Validation", device)
        m_val["n"] = sum(m["per_class_total"].values())
        return float(m["balanced"])

    game = CachedGame(n_players=N_PLAYERS, evaluate=evaluate)
    t0 = time.perf_counter()
    v_full = game(FULL_MASK)
    m = m_val.get("n") or len(val_loader.dataset)
    threshold = math.sqrt(max(v_full * (1.0 - v_full), 1e-12) / m)
    rungs, rung1, rung2, ties = backward_ladder(game, threshold=threshold)
    game_seconds = time.perf_counter() - t0
    print(f"balanced ladder rung1 k={rung1['k']} rung2 k={rung2['k']} "
          f"floor={threshold:.4f} evals={game.calls}")

    inner.aggregate_inter_nbhd = wrapper._original_aggregate
    model_r2 = copy.deepcopy(model)
    rung1.update(continue_and_test(
        DS, args.seed, model, pipe.datamodule, rung1["mask"],
        run_dir / "continue_rung1", device, model_name=model_name))
    rung2.update(continue_and_test(
        DS, args.seed, model_r2, pipe.datamodule, rung2["mask"],
        run_dir / "continue_rung2", device, model_name=model_name))

    # balanced val/test for both trained candidates; select on balanced val
    trained = {"rung1": model, "rung2": model_r2}
    for name, rung in (("rung1", rung1), ("rung2", rung2)):
        vm = split_metrics(trained[name], pipe.datamodule.val_dataloader(),
                           "Validation", device)
        tm = split_metrics(trained[name], pipe.datamodule.test_dataloader(),
                           "Test", device)
        rung["val_balanced_accuracy"] = float(vm["balanced"])
        rung["test_balanced_accuracy"] = float(tm["balanced"])
    selected = ("rung2" if rung2["val_balanced_accuracy"]
                > rung1["val_balanced_accuracy"] else "rung1")
    winner = rung1 if selected == "rung1" else rung2

    row = {
        "dataset": DS, "seed": args.seed, "selector": "ladder_b2_balanced",
        "rung1": rung1, "rung2": rung2, "selected": selected,
        "mask": int(winner["mask"]), "k": int(winner["k"]),
        "val_accuracy": float(winner["val"]),
        "test_accuracy": float(winner["test"]),
        "val_balanced_accuracy": winner["val_balanced_accuracy"],
        "test_balanced_accuracy": winner["test_balanced_accuracy"],
        "ladder": rungs, "ties": ties, "v_full_balanced": float(v_full),
        "balanced_floor": float(threshold), "m_val": int(m),
        "floor_note": "binomial-form floor applied to balanced accuracy; "
                      "approximation, labeled as such",
        "n_game_evaluations": int(game.calls),
        "stem_seconds": float(stem_seconds),
        "game_seconds": float(game_seconds),
        "continue_seconds_rung1": rung1["continue_seconds"],
        "continue_seconds_rung2": rung2["continue_seconds"],
        "stem_epochs": int(args.stem_epochs), "model": args.model,
        "patches": PATCHES,
    }
    with open(out_path, "w") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"wrote {out_path}: selected={selected} k={row['k']} "
          f"test_balanced={row['test_balanced_accuracy']:.4f}")


if __name__ == "__main__":
    main()
