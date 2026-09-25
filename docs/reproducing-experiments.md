# Reproducing the paper's experiments

Every experiment has one **runner** and one **assembler** under
`experiments/`. Runners take seeds and output locations on the command
line, write one JSONL per (dataset, seed) under `results/<arm>/`, and
are resumable (existing outputs are skipped, so array jobs can be
resubmitted safely). Assemblers fold completed runs into the saved
frames under `data/frozen/`. Two studies were produced on a separate
cluster and ship as saved numerical results instead of runners — see
[data/README.md](../data/README.md).

## Environment for every run

```bash
export TOPOSHAP_INTERRANK_ORIENTATION=fixed
export TOPOSHAP_GRAPHXAI_DATA=/path/to/graphxai/datasets   # GraphXAI runs only
```

Training seeds are 42–44 everywhere, extended to 42–46 for the GCCN
B=2 selection; the sampled-attribution estimator uses its own seed 0.
Reported runs used one NVIDIA H200 GPU each; any recent CUDA GPU works.
Runners are plain Python scripts — submit them under your scheduler
however you normally would, one (dataset, seed) cell per GPU.

## Code-to-paper terminology

The code predates the paper's final wording. The mapping:

| Code name | Paper term |
| --- | --- |
| `ladder_b2`, "ladder" | the $B{=}2$ selection pipeline (backward elimination over masked validation accuracy) |
| "stem", `stem_epochs` | the briefly trained full model (five epochs) |
| "rung", `rung1` | a shortlisted candidate; `rung1` is the stop candidate |
| "vocabulary", `full_vocabulary` | the eleven candidate neighborhoods; the all-neighborhood model |
| `autok` | the $B{=}1$ arm (stop candidate only) |
| `cheap_v` | masked validation accuracy $a(S)$ along the elimination path |
| "landscape" | one exhaustive benchmark setting (all $2^6$ subsets trained separately) |
| "menu" | the published-architecture baseline |

## Explaining predictions (paper Section 5)

| Experiment | Train | Explain / evaluate |
| --- | --- | --- |
| Graph benchmarks (GEA, GCN/GIN) | `train_graph_gnns.py` | `molecule_participation_gea.py` (attribution output), `molecule_select_gea.py` + `select_on_gnn.py` (explanation output); both record wall-clock per instance |
| Baseline explainers (SubgraphX, GNNExplainer, PGExplainer) | — | shipped as saved results (`data/frozen/baselines2/`); `run_graph_baselines.py` reruns them against the same models and splits |
| Lifted complexes (both liftings) | `train_molecule_topk11.py` (standard), `train_molecule_topk11s.py` (structure-only) | `molecule_select_gea.py` / `molecule_participation_gea.py` on the k11 / k11s arms; the same outputs carry the face-retention / boundary-recall compositions and the recomputed-features rescoring |
| Faithfulness without annotations (deletion-AUC) | `train_structural_nogt.py` | `nogt_usefulness.py` (TopoSHAP, occlusion, gradient-times-input, random on the same fixed instance draw) |
| MANTRA attribution pair and AUROCs | selection runners below | `mantra_attribution_figure.py`, `mantra_attrfig_hopse_rerun.py` (per-cell games at 512 orderings), `mantra_matched_pairs.py` (count-conditional AUROC) |

## Explaining performance (paper Section 6)

| Experiment | Runner |
| --- | --- |
| GCCN selection, $B{=}2$ (headline) | `run_ladder_b2.py` (`--stem-epochs 3/5/10` gives the initial-training sensitivity arms) |
| HOPSE selection, $B{=}2$ | `run_ladder_b2_hopse.py` (sharded encodings: `merge_hopse_shards.py`) |
| $B{=}1$ arm | `run_autok.py` |
| GCCN sweep | `run_gccn_sweep.py` |
| HOPSE sweep | `run_hopse_sweep.py` |
| Published-architecture baseline (MANTRA) | `run_gccn_menu.py` |
| Shortlist controls (random $B{=}2$, full-model default) | `run_baseline_shortlists.py` |
| Balanced-selection MANTRA variant | `run_ladder_b2_balanced.py`, evaluated by `mantra_balanced_eval.py` |
| FLOPs accounting | `flops_profiler.py` (in-run), `profile_b2_flops.py` (reconstruction for runs recorded before the profiler) |
| Exhaustive six-neighborhood benchmark | shipped as saved results (`data/frozen/pooled.json`); `run_landscape.py` trains all 64 subsets for one setting if you want to extend it |

Assemblers: `assemble_ladder_b2.py`, `assemble_autok.py`,
`assemble_hopse_sweep.py`, `assemble_gccn_sweep.py`,
`assemble_hopse_k3.py`, `assemble_mantra.py`, `assemble_anchored3.py`,
`assemble_campaign.py`.

The attribution-agreement and oracle analyses of the paper's controlled
evaluation are deterministic functions of `data/frozen/pooled.json`
(the per-setting seed-mean values of all 64 subsets and the masked-game
attribution profiles); the paper's appendix defines both statistics
exactly.

## Consistency checks

`experiments/` also carries the guards used during the campaigns:
construction-time full-coalition fidelity checks live in the games
themselves; `hopse_gates.py`, `hopse_noop_gate.py`, `autok_gate_c.py`,
and `verify_marker_arm.py` re-verify masking and selection invariants
against recorded runs; `resume_guard.py` is the shared
skip-if-complete logic.
