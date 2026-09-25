# Saved numerical results

`data/frozen/` carries the saved results that back the paper's
analyses. Files here are read through `toposhap.io` and are treated as
read-only: reruns write to `results/` first and are folded in by the
assemblers under `experiments/`. Provenance metadata inside the shipped
copies (machine hostnames, filesystem paths, experiment-tracker IDs) is
redacted; every numerical value is untouched.

## Record-produced studies (shipped as data, no local runner)

Two studies were produced on a separate cluster and ship as saved
results:

- `pooled.json` — the exhaustive six-neighborhood benchmark: for each
  of the twelve dataset/hyperparameter settings, the seed-mean test
  values of all $2^6 = 64$ separately trained neighborhood subsets
  (three seeds), together with the masked-game attribution profiles
  used by the attribution-agreement analysis and the dataset/metric
  labels.
- `baselines2/` — the GraphXAI explainer suite (SubgraphX,
  GNNExplainer, PGExplainer, random) on the shared GCN/GIN models:
  per-instance GEA/GEF records plus `manifest.json` documenting the
  patched GraphXAI commit, explainer settings, model training settings,
  and dataset statistics.

## Selection and attribution frames (regenerable from runners)

The `*.parquet` frames are the assembled outputs of the local
experiment runners — one row per (dataset, seed) with test metrics and
FLOPs: `ladder_b2_*` ($B{=}2$ selections for GCCN and HOPSE),
`autok_marker` ($B{=}1$ arm), `baseline_markers` (sweep winners and
other baseline markers), `hopse_sweep_*` and `gccn_menu_mantra`, and
`anchored3_exact*` (exact neighborhood-attribution games).
`mantra_attr_fig.json` / `mantra_attr_hopse.json` hold the per-cell
attribution games behind the MANTRA count-matched pair. Rerunning the
corresponding runner and assembler (see
[docs/reproducing-experiments.md](../docs/reproducing-experiments.md))
reproduces any of these.
