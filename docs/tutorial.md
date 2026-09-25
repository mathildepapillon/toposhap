# Tutorial: explaining TNN predictions and architectures

This tutorial walks through both TopoSHAP games on a toy complex you can
run on CPU in seconds, then shows the pattern for real trained TopoBench
models. Install per the [README](../README.md) first. Every snippet in
sections 1–3 is executable as-is.

Setup that every script needs once:

```python
import os, torch
os.environ["TOPOSHAP_INTERRANK_ORIENTATION"] = "fixed"
from toposhap.patches import apply_all; apply_all()
```

The environment variable selects the patched GCCN inter-rank message
orientation, and `apply_all()` installs the TopoBench guards (see
[topobench-integration.md](topobench-integration.md)).

## 1. The cell game: which cells drove a prediction?

Players are the cells of the input complex, at every rank. A coalition
$S$ keeps a subset of cells; every cell outside $S$ is **detached** —
its feature row is zeroed, every neighborhood matrix is rebuilt without
it, and it is excluded from readout pooling. The game value compares
the model's output on the masked complex with its output on the empty
complex.

```python
from toposhap.cells import CellPlayer
from toposhap.cells.participation import CellParticipationGame
from toposhap.cells.select import greedy_prune_schedule
from toposhap.shapley.exact import shapley_values
from toposhap.shapley.games import CachedGame
from toposhap.testing import full_vocabulary_backbone, two_triangle_participation_batch

# Toy fixture: a rank-2 complex with 4 nodes, 5 edges, 2 faces, and a
# GCCN backbone carrying all eleven candidate neighborhoods.
backbone = full_vocabulary_backbone(seed=6)
batch = two_triangle_participation_batch(seed=7)

def model_fn(b):                       # the scalar being explained
    with torch.no_grad():
        return backbone(b)[0].sum()    # readout over the node rank

players = ([CellPlayer(rank=0, index=i) for i in range(4)]
           + [CellPlayer(rank=1, index=i) for i in range(5)]
           + [CellPlayer(rank=2, index=i) for i in range(2)])

raw = CellParticipationGame(model_fn, batch, players)
game = CachedGame(n_players=len(players), evaluate=raw)
```

`CellParticipationGame` verifies at construction that the rebuilt
full-coalition input reproduces the model's stored forward pass exactly
— if your model or batch violates the masking assumptions, you find out
immediately rather than in the numbers.

**Output 1 — attributions.** With 11 players, all $2^{11}$ coalitions
are affordable, so the Shapley values are exact:

```python
phi = shapley_values(game, len(players))     # one value per cell
full = (1 << len(players)) - 1
assert abs(phi.sum() - (game(full) - game(0))) < 1e-5   # efficiency axiom
```

**Output 2 — compact explanations.** Greedy backward pruning removes,
at each step, the cell whose removal lowers the value least, returning
one nested candidate per size:

```python
schedule = greedy_prune_schedule(game, len(players), budgets=range(1, len(players)))
best3, value3 = schedule[3]     # frozenset of 3 player indices + its value
```

Coalition values are cached by bitmask (`CachedGame`), so the pruning
pass reuses everything the Shapley computation already evaluated.

### Scaling up: the sampled estimator

Exact enumeration stops being affordable around 20 players. Beyond
that, use the unbiased permutation-sampling estimator; its budget
`passes` counts every evaluated player ordering (orderings are drawn
uniformly at random, each followed by its reverse), and the cost is at
most `passes * n + 1` model evaluations, fewer with the cache:

```python
from toposhap.shapley.sampled import sampled_shapley
att = sampled_shapley(game, len(players), passes=128, seed=0)
att.phi        # point estimates
att.stderr     # per-player standard error over orderings
```

### The feature-masking variant and interaction indices

`toposhap.cells.explain` implements a second game in which a masked
cell keeps its wiring and only its *features* are replaced. It answers
"whose signal matters" rather than "whose presence matters", and it is
the variant that supports pairwise Shapley interaction indices:

```python
from toposhap.cells.explain import explain_cells
from toposhap.testing import two_triangle_batch

expl = explain_cells(model_fn, two_triangle_batch(seed=0),
                     players[:4] + players[-2:], interactions=True)
expl.phi             # attributions under feature masking
expl.interactions    # pairwise interaction index per cell pair
```

The paper's headline results use the detachment game of section 1;
prefer it unless you specifically want fixed-wiring semantics.

## 2. The neighborhood game: which neighborhoods drive performance?

Players are the architecture's neighborhoods, in the frozen bit order
of `toposhap.vocabulary.NEIGHBORHOODS`. `CoalitionMaskedBackbone` wraps
a GCCN backbone so that a bitmask silences the message-passing routes
outside the coalition — one forward pass per coalition, no retraining:

```python
from toposhap.neighborhoods import CoalitionMaskedBackbone
from toposhap.selection.selectors import top_k_mask
from toposhap.vocabulary import NEIGHBORHOODS

wrapped = CoalitionMaskedBackbone(full_vocabulary_backbone(seed=6))

def score(mask: int) -> float:
    # TopoTune's forward mutates its input batch: build a fresh one per call.
    batch = two_triangle_participation_batch(seed=7)
    with wrapped.coalition(mask), torch.no_grad():
        return float(wrapped(batch)[0].sum())

n = len(NEIGHBORHOODS)                          # 11 -> 2048 coalitions, cached
ngame = CachedGame(n_players=n, evaluate=lambda m: score(m) - score(0))
phi_neighborhood = shapley_values(ngame, n)
mask3 = top_k_mask(phi_neighborhood, 3)         # bitmask of the top-3 players
```

On a real dataset the score is the **masked validation accuracy** of a
briefly trained (five-epoch) full model, and the empty coalition — the
model with every neighborhood masked — is the baseline. Two caveats
carry over from the paper: masked attributions describe reliance in
that trained model, not which neighborhoods a separately trained model
would need, and Shapley values are not a sound selector when two
neighborhoods carry the same information. The selection pipeline
therefore scores candidate *removals* directly.

### Architecture selection

The full pipeline (brief training, backward elimination over masked
validation accuracy, threshold-marked stop candidate, $B=2$ shortlist
continued to full training, validation pick) is implemented in
`experiments/run_ladder_b2.py` against real TopoBench datasets, with a
HOPSE variant in `run_ladder_b2_hopse.py`. To prune a backbone to a
selected coalition permanently (dropping the excluded modules rather
than masking them), use:

```python
from toposhap.neighborhoods import prune_backbone_
removed = prune_backbone_(backbone, mask3)   # in place; returns removed routes
```

## 3. Explaining a trained TopoBench model

The pattern is identical to section 1 with two substitutions: the model
function reads a fixed target logit, and the players enumerate the real
complex's cells. For a hydra-instantiated TopoBench `TBModel` with a
sum-pool readout, the bundled helpers build the model function:

```python
from toposhap.cells.participation import (
    CellParticipationGame, pool_faithful_logit, pool_faithful_margin)

model.eval()
model_fn = pool_faithful_logit(model, target_cls=label)   # one fixed class logit
players = ([CellPlayer(rank=0, index=i) for i in range(n_nodes)]
           + [CellPlayer(rank=1, index=i) for i in range(n_edges)]
           + [CellPlayer(rank=2, index=i) for i in range(n_faces)])
game = CachedGame(n_players=len(players),
                  evaluate=CellParticipationGame(model_fn, batch, players))
```

Conventions from the paper: the target logit is held fixed across all
coalitions of an instance (annotated class on labeled benchmarks,
predicted class otherwise, orientable-class margin via
`pool_faithful_margin` on MANTRA); explanation experiments use the
sampled estimator with `passes=128` and estimator seed 0.
`experiments/molecule_participation_gea.py` and
`experiments/nogt_usefulness.py` are complete, working examples.

**HOPSE models** consume precomputed positional/structural encodings,
so plain feature masking never reaches their computation. Use
`HopseParticipationGame`, which recomputes the encodings on every
masked complex and additionally checks that recomputed full-coalition
encodings reproduce the stored prediction:

```python
from toposhap.cells.participation import HopseParticipationGame
```

## 4. Baselines and metrics

`toposhap.baselines` provides occlusion (leave-one-out drop under the
same game) and gradient-times-input on raw features;
`toposhap.metrics` provides GEA (Jaccard against an annotated motif),
GEF, and deletion curves / deletion-AUC (`toposhap.metrics.deletion`),
matching the paper's evaluation protocols.
