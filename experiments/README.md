# Experiment runners

One runner and one assembler per experiment. Runners write one JSONL
per (dataset, seed) under `results/<arm>/` and skip cells whose output
already exists, so interrupted campaigns resume by resubmitting the
same command. Assemblers fold completed runs into the saved frames
under `data/frozen/`.

The full experiment-to-runner map, the required environment variables,
seeds, and the code-to-paper terminology table are in
[docs/reproducing-experiments.md](../docs/reproducing-experiments.md).

## Running

Every runner is a plain Python script:

```bash
export TOPOSHAP_INTERRANK_ORIENTATION=fixed
python experiments/run_ladder_b2.py --dataset benzene --seed 42
```

Datasets download through TopoBench's loaders on first use, except the
GraphXAI datasets, which need `TOPOSHAP_GRAPHXAI_DATA` pointing at a
GraphXAI data directory. One (dataset, seed) cell fits on one GPU; under
a batch scheduler, submit one cell per task, e.g.

```bash
#SBATCH --gres=gpu:1
python experiments/run_ladder_b2.py --dataset "$DATASET" --seed "$SEED"
```

`topobench_driver.py` is the shared bridge that composes TopoBench
hydra configs, trains, and evaluates with the repository's conventions
(default TopoBench optimizer settings; early stopping patience 25;
validation-best checkpoints). `resume_guard.py` is the shared
skip-if-complete logic.
