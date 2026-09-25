# TopoSHAP

Game-theoretic explanations for topological deep learning (TDL).
TopoSHAP explains the two questions a TNN practitioner asks most:

1. **Which cells drove this prediction?** `TopoSHAP-cell` treats the
   cells of a combinatorial complex — nodes, edges, faces — as players
   in a cooperative game. Detaching the cells outside a coalition gives
   a masked complex; Shapley values over that game attribute the
   prediction to individual cells across ranks, and greedy pruning
   returns one compact explanation per size. On plain graphs the game
   reduces to induced-subgraph removal, so the method is directly
   comparable to GNN explainers.
2. **Which neighborhoods drive performance?** `TopoSHAP-neighbor`
   treats a TNN's message-passing neighborhoods as players. Masking a
   briefly trained full model scores each coalition by validation
   accuracy in one forward pass, which supports per-neighborhood
   performance attribution and a cheap architecture-selection pipeline:
   backward elimination proposes candidate architectures, a small
   shortlist ($B=2$) is trained to completion, and validation accuracy
   picks the winner.

Both games run on [TopoBench](https://github.com/geometric-intelligence/TopoBench)
models (GCCN / TopoTune and HOPSE families) without modifying them:
everything works through forward evaluations.

## Installation

Python 3.11, PyTorch 2.3.0 (CUDA 12.1), torch-geometric with matching
scatter/sparse wheels, and TopoBench pinned at commit `6d8953e7e170`:

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
uv pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
uv pip install torch-scatter torch-sparse torch-cluster --no-build-isolation \
  -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
# TopoBench is not vendored -- clone the pinned commit
# (uv venvs have no setuptools; --no-build-isolation needs it):
git clone https://github.com/geometric-intelligence/TopoBench external/TopoBench
git -C external/TopoBench checkout 6d8953e7e170
uv pip install setuptools wheel
uv pip install -e external/TopoBench --no-build-isolation
pytest   # the core suite passes without GPUs or downloaded datasets
```

Plain `pip` works the same way (create the venv with
`python3.11 -m venv`, drop the `uv` prefixes).

Two environment conventions apply to every run:

```bash
export TOPOSHAP_INTERRANK_ORIENTATION=fixed   # patched GCCN inter-rank semantics
export TOPOSHAP_GRAPHXAI_DATA=/path/to/graphxai/datasets   # GraphXAI runs only
```

and Python code should apply the bundled TopoBench guards once at
startup: `from toposhap.patches import apply_all; apply_all()` (see
[docs/topobench-integration.md](docs/topobench-integration.md) for what
they fix).

## Quickstart: explain a prediction

Cells are players; a cell outside the coalition is *detached* — its
feature row is zeroed, every neighborhood matrix is rebuilt without it,
and it drops out of readout pooling.

```python
import os, torch
os.environ["TOPOSHAP_INTERRANK_ORIENTATION"] = "fixed"
from toposhap.patches import apply_all; apply_all()

from toposhap.cells import CellPlayer
from toposhap.cells.participation import CellParticipationGame
from toposhap.cells.select import greedy_prune_schedule
from toposhap.shapley.exact import shapley_values
from toposhap.shapley.games import CachedGame
from toposhap.testing import full_vocabulary_backbone, two_triangle_participation_batch

# A toy rank-2 complex (4 nodes, 5 edges, 2 faces) and a GCCN carrying
# all eleven candidate neighborhoods.
backbone = full_vocabulary_backbone(seed=6)
batch = two_triangle_participation_batch(seed=7)

def model_fn(b):                       # the scalar being explained
    with torch.no_grad():
        return backbone(b)[0].sum()    # readout over the node rank

players = ([CellPlayer(rank=0, index=i) for i in range(4)]
           + [CellPlayer(rank=1, index=i) for i in range(5)]
           + [CellPlayer(rank=2, index=i) for i in range(2)])

# Output 1: exact Shapley attributions under detachment semantics.
game = CachedGame(n_players=len(players),
                  evaluate=CellParticipationGame(model_fn, batch, players))
phi = shapley_values(game, len(players))     # one value per cell

# Output 2: one compact explanation per size via greedy pruning.
schedule = greedy_prune_schedule(game, len(players), budgets=range(1, len(players)))
best3, value3 = schedule[3]                  # the size-3 candidate coalition
```

With more than ~20 players, swap `shapley_values` for the sampled
estimator (`toposhap.shapley.sampled.sampled_shapley`, unbiased,
`passes` evaluated player orderings). For a *trained* TopoBench model
the pattern is identical with `model_fn` returning the target-class
logit; HOPSE models need `HopseParticipationGame`, which recomputes the
positional/structural encodings on every masked complex. The
[tutorial](docs/tutorial.md) walks through all of this end to end,
including the feature-masking game variant (`toposhap.cells.explain`)
and pairwise interaction indices.

## Quickstart: attribute performance to neighborhoods

```python
from toposhap.neighborhoods import CoalitionMaskedBackbone
from toposhap.shapley.exact import shapley_values
from toposhap.shapley.games import CachedGame
from toposhap.vocabulary import NEIGHBORHOODS   # bit i = player i, frozen order

wrapped = CoalitionMaskedBackbone(full_vocabulary_backbone(seed=6))

def score(mask: int) -> float:
    # TopoTune's forward mutates its input batch: build a fresh one per call.
    batch = two_triangle_participation_batch(seed=7)
    with wrapped.coalition(mask), torch.no_grad():
        return float(wrapped(batch)[0].sum())

n = len(NEIGHBORHOODS)                          # 11 -> 2048 coalitions, cached
game = CachedGame(n_players=n, evaluate=lambda m: score(m) - score(0))
phi_neighborhood = shapley_values(game, n)
```

In the paper's pipeline the score is the masked **validation accuracy**
of a five-epoch model, backward elimination walks from all eleven
neighborhoods down to one, and the two shortlisted candidates continue
training under the dataset's standard settings —
`experiments/run_ladder_b2.py` is the complete implementation and
[docs/reproducing-experiments.md](docs/reproducing-experiments.md) maps
every experiment to its runner.

## Using TopoSHAP inside TopoBench

This repository is the standalone research package. The method is also
integrated into TopoBench itself as `topobench.explain` (games, exact +
sampled Shapley, cell explanations, neighborhood masking and selection,
axiom checks, tests, and a notebook tutorial), generalized so the
player order is whatever `neighborhoods` list the backbone was built
with. See [docs/topobench-integration.md](docs/topobench-integration.md)
for the integration branch, the pinned commit, and the three upstream
bug fixes the patches address.

## Repository layout

```
src/toposhap/
  vocabulary.py      the 11 candidate neighborhoods; bit order = player order
  shapley/           exact values + interaction indices; sampled estimator; game caching
  cells/             cell games (detachment + feature-masking), greedy explanations
  neighborhoods/     coalition masking / pruning of GCCN backbones; brief-training recipe
  selection/         top-k / anchored / greedy selectors; automatic-k; submodularity census
  metrics/           GEA / GEF / deletion curves; axioms as executable checks
  costs/             FLOPs accounting (FlopCounterMode + eigh/spmm pricing)
  baselines.py       occlusion and gradient-times-input under the same games
  patches/           TopoBench guards (orientation fix switch, batch snapshot/restore)
  io/                loaders for the saved numerical results
tests/               exactness vs brute force, masking consistency, axiom checks
experiments/         one runner + one assembler per experiment (see experiments/README.md)
docs/                tutorial, experiment reproduction guide, TopoBench integration
data/frozen/         saved numerical results shipped with the repo (see data/README.md)
```

## Reproducing the paper's experiments

Each experiment has one runner and one assembler under `experiments/`,
with seeds and output locations set on the command line; runners are
resumable and write one JSONL per (dataset, seed). Two studies were
produced on a separate cluster and ship as saved numerical results
instead (the exhaustive six-neighborhood benchmark and the GraphXAI
explainer suite; see [data/README.md](data/README.md)). The full map
from paper experiment to runner, including seeds, hardware, and the
code-to-paper terminology table, is
[docs/reproducing-experiments.md](docs/reproducing-experiments.md).

## Tests

```bash
TOPOSHAP_INTERRANK_ORIENTATION=fixed pytest
```

The suite includes the exactness check of the Shapley implementation
against brute-force enumeration (machine precision), full-coalition
fidelity guards (the rebuilt masked input must reproduce the stored
forward pass), attribution invariance under batch permutation, and
equality on provably symmetric cells. Tests that need the pinned
TopoBench or the saved results skip cleanly when those are absent.

## License and citation

MIT (see [LICENSE](LICENSE)). A citation entry will be added upon
publication of the accompanying paper.
