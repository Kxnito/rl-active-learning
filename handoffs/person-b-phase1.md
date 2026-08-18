# Person B Phase 1 — RL Environment & Agent

## Status

✅ **Phase 1 Complete** (walking skeleton + first real comparison run)

This component provides the Gymnasium environment, reward design, and Stable-Baselines3 (MaskablePPO) agent for the RL active learning project, plus the evaluation glue connecting it to Person A's baselines and Person C's `eval/` tooling.

The full loop has been run end-to-end, including a real 5-seed comparison against the baselines:

```text
data/dataset.py (Person A)
    ↓
DatasetSplits
    ↓
env/active_learning_env.py — Gymnasium env wrapping a sklearn LogisticRegression "student model"
    ↓
agent/train.py — MaskablePPO training (sb3-contrib, action-masked)
    ↓
agent/evaluate.py — deterministic rollout of the trained policy
    ↓
eval/run_experiment.py — combines RL curve with Person A's baseline curves
    ↓
local-test/results/experiments/*.csv
    ↓
eval/compare_methods.py + eval/plot_learning_curves.py (Person C)
```

---

## What This Component Owns

### RL Environment

* Custom Gymnasium environment (`ActiveLearningEnv`)
* State/action/reward definitions (`project-context.md` Section 1)
* Fixed-size masked action space (handles the shrinking-pool problem)
* Student model retraining loop, feature scaling

### RL Agent

* MaskablePPO training setup (`agent/train.py`)
* Trained-policy evaluation rollout (`agent/evaluate.py`)

### Evaluation Glue

* `eval/run_experiment.py` — the script that actually produces comparable results across random/uncertainty/RL

---

## Key Files

| File | Purpose |
|---|---|
| `env/active_learning_env.py` | The Gymnasium environment — `reset()`, `step()`, `action_masks()`, `_get_obs()` |
| `agent/train.py` | `train_agent(splits, budget, total_timesteps)` — builds the masked env and trains MaskablePPO |
| `agent/evaluate.py` | `evaluate_agent(model, splits, budget)` — deterministic rollout, returns `(curve, final_test_accuracy)` |
| `eval/run_experiment.py` | Runs random sampling, uncertainty sampling, and the RL agent on the same dataset/seed/budget; writes one combined CSV in `eval/`'s result schema |
| `tests/test_env.py` | 15 tests covering `reset()`/`step()`/`action_masks()` happy paths and edge cases |

---

## How to Run It

```bash
# Run the test suite
python -m pytest tests/ -W ignore::RuntimeWarning

# Quick smoke test — confirms the training loop runs without crashing
python -c "
from data.dataset import load_dataset
from agent.train import train_agent
splits = load_dataset(seed_size=20, val_size=100, test_size=100)
model = train_agent(splits, budget=50, total_timesteps=200, verbose=1)
"

# Full experiment — one seed
python -m eval.run_experiment --seed 42

# Multiple seeds (what the current results are based on)
for seed in 42 43 44 45 46; do python -m eval.run_experiment --seed $seed; done

# Generate the comparison summary + plot
python -m eval.compare_methods --input-dir local-test/results/experiments --output-dir outputs/evaluation
python -m eval.plot_learning_curves --input-dir local-test/results/experiments --output outputs/plots/learning_curves.png --dataset breast_cancer
```

Note: `eval/run_experiment.py` must be run as a module (`python -m eval.run_experiment`), not directly (`python eval/run_experiment.py`) — the latter doesn't put the repo root on `sys.path`, so the `data`/`env`/`agent` imports fail.

---

## Design Decisions & Why

**Masked fixed-size action space, not a shrinking one.** SB3's action spaces are fixed for the lifetime of training, but the unlabeled pool shrinks by one every step as labels get revealed. Fix: `Discrete(pool_capacity)` over the *original* pool size, with already-revealed indices marked invalid via `action_masks()`. This requires `MaskablePPO` from `sb3-contrib` — plain SB3 DQN/PPO don't support action masking at all.

**`StandardScaler` fit once on `seed_X + pool_X`, not refit per episode.** Originally fit fresh on just the 20-sample seed set each `reset()` — unstable, since at least one of 30 features had near-zero variance in such a small sample, producing scaled values 15+ std devs out on the wider pool and overflowing the model's matmul. Fitting once on the full feature pool up front isn't a label leak (only `pool_y` is meant to stay hidden, not `pool_X`) and is stable. **This exact bug also existed in `data/baseline.py`'s `StudentModel` and was fixed the same way** — worth knowing if anyone else writes a student-model variant later.

**Reward = val accuracy delta.** `reward_t = val_accuracy(model_t) - val_accuracy(model_{t-1})`, per `project-context.md` Section 8. Computed on a held-out validation split, never touched by training.

**3-element observation vector:** pool uncertainty (`1 - mean(max predicted probability)`), labels-used fraction (`num_revealed / budget`), and class balance among revealed labels. This was a starting design, not a proven-good one — see Known Issues.

**`labels_used` counts seed + revealed, not just revealed.** `agent/evaluate.py` originally only counted `Oracle.num_revealed` (pool queries), which didn't match `data/baseline.py`'s convention of seed-size + queries. Caught by literally reading the output CSV and noticing the RL curve's `labels_used` column started at 1 instead of 21 like the baselines — fixed to match, since otherwise the x-axes in every plot silently wouldn't line up.

---

## Known Issues / Things I Didn't Get To

* **The RL agent is not yet clearly beating uncertainty sampling.** 5-seed result: RL and uncertainty are statistically tied on final val accuracy (0.986 mean each); RL edges ahead slightly on normalized AUC (0.973 vs 0.970) but that's within noise at this sample size. RL's real, measurable advantage right now is *consistency* — its std across seeds (0.0055) is under half of uncertainty's (0.0114). Both clearly beat random (0.964).
* **No reward/observation tuning has happened yet.** The 3-element state vector was never iterated on — it's plausible a richer state (e.g. per-class uncertainty, model confidence trend) would help the agent actually outperform uncertainty sampling instead of just matching it.
* **Never run on SageMaker.** Everything above is local only. Person C's infra (`infra/launch_training_job.py`, `infra/training/train.py`) is built and tested, but per their handoff doc it's still only running an infrastructure smoke test, not the real `env`/`agent` logic.
* **No tests for `agent/evaluate.py` or `eval/run_experiment.py`.** `tests/test_env.py` covers the environment itself well, but the evaluation rollout and the CSV-writing glue are untested.
* **Only tested on Breast Cancer Wisconsin.** `project-context.md` Section 8 names Adult Census Income as the flagship second dataset — never attempted.
* **The GitHub Actions CI question was never fully resolved.** At one point PRs #7 and #8 merged without triggering any workflow run at all (no error, no pending state — just silence), for reasons that were never pinned down (ruled out: disabled workflow, outside-contributor approval gating, missing workflow file). Worth another look — whoever has repo admin access should check Settings → Actions → General → "Actions permissions" specifically.

---

## Gotchas

* **macOS + Apple Accelerate BLAS produces spurious `RuntimeWarning: divide by zero encountered in matmul` warnings.** Confirmed harmless (checked for NaN/Inf directly — none; probabilities are valid and sum to 1.0). Tied to `numpy`'s BLAS backend on this specific machine, not a real numerical bug. Filter with `-W ignore::RuntimeWarning` if it gets noisy; don't chase it as a bug.
* **Editing a `.py` file you've already `import`ed in a live REPL session doesn't take effect until you restart the REPL.** Bit everyone at least once this project — Python caches the imported class/function definitions in memory; the file on disk changing doesn't retroactively update them.
* **Real experiment sweep output must go in `local-test/results/experiments/`, not `local-test/results/` directly.** The parent directory also holds Person C's hand-crafted example fixtures (`random-seed42.csv`, `uncertainty-seed42.csv`), which use the same `run_id` convention (`method-seed`). `eval/load_results.py` globs recursively and concatenates everything it finds — real output landing next to the examples silently merges under matching `run_id`s and corrupts `compare_methods.py`'s aggregation (this actually happened once — caught it by noticing the numbers didn't add up).
* **`Oracle` doesn't validate that `env`'s `LogisticRegression`/scaler assumptions hold for other datasets.** If someone points this whole pipeline at Adult Census Income (larger, class-imbalanced, mixed categorical/numeric features), expect the scaling/convergence story to need revisiting — it was tuned against Breast Cancer Wisconsin's specific shape (569 samples, 30 continuous features).

---

## What I'd Do Next If I Kept Working On This

1. Tune the observation vector / reward design and re-run the seed sweep to see if it actually closes the gap with uncertainty sampling.
2. Add `tests/test_evaluate.py` and `tests/test_run_experiment.py`.
3. Wire `agent/train.py`/`agent/evaluate.py` into Person C's SageMaker pipeline instead of running locally.
4. Run the same 5-seed comparison on Adult Census Income.
5. Chase down the GitHub Actions mystery properly — it means nothing's been auto-verifying PRs for a while.
