# Using TopoSHAP with TopoBench

TopoSHAP runs against [TopoBench](https://github.com/geometric-intelligence/TopoBench)
models through forward evaluations only, so message-passing GCCNs
(TopoTune) and non-message-passing HOPSE models are treated
identically. This page covers the pinned commit, the bundled patches,
and the native integration branch.

## The pin

All experiments use TopoBench at commit `6d8953e7e170`, installed
editable under `external/TopoBench` (see the README's installation
block). The pin matters: the games verify at construction that a
rebuilt full-coalition input reproduces the model's stored forward
pass, and that guarantee is only tested against this commit.

## The bundled patches

`toposhap.patches.apply_all()` installs three guards at runtime,
without modifying the TopoBench checkout:

1. **Inter-rank orientation.** The pinned GCCN builds the inter-rank
   `edge_index` with an inconsistent source/destination orientation on
   some neighborhoods. With `TOPOSHAP_INTERRANK_ORIENTATION=fixed`
   (required for every run in this repo), messages flow source-to-
   destination as the neighborhood's definition states. The unpatched
   behaviour remains selectable (`upstream`) only so the bug's effect
   can be reproduced and studied.
2. **Batch mutation guard.** TopoTune's forward overwrites `x_{rank}`
   on its input batch. Repeated coalition evaluations therefore
   snapshot and restore batch features and neighborhood attributes;
   the games do this automatically, and the patch protects direct use.
3. **Full-coalition fidelity check.** At construction, every game
   evaluates the full coalition and requires it to match the model's
   stored forward pass exactly (HOPSE games additionally check that
   recomputed encodings reproduce the stored prediction within
   tolerance), so a masking/model mismatch fails loudly.

## The native integration branch

For TopoBench users who want the method without this repository, the
integration branch `toposhap-upstream` (based on the same pinned
commit) carries:

- `topobench/explain/` — the games, exact Shapley values and
  interaction indices, the sampled estimator, cell-masking
  explanations, neighborhood coalition masking and pruning, selectors
  with automatic-$k$, and axiom checks as executable tests. The player
  order is generalized to whatever `neighborhoods` list the backbone
  was configured with, not a fixed set of eleven.
- The inter-rank orientation fix applied in place (no environment
  switch), with a regression test.
- `test/explain/` — exactness against brute force at machine
  precision, masking equivalence against subset-built backbones,
  selector behaviour.
- `tutorials/tutorial_explain.ipynb` — a notebook walking from a
  hydra-instantiated TopoBench model to attributions.

The orientation fix is a plain correctness fix and is valuable
upstream independently of TopoSHAP; it is sequenced as the first
upstream pull request.
