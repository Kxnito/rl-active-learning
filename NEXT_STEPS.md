# Next Steps — Person A / B / C

*Compiled from `handoffs/person-c-phase1.md` and `handoffs/person-b-phase1.md`. Update this as items get done — check them off or delete them, don't let it go stale.*

---

## Person A — Data + Baseline

- [ ] **Add `tests/test_baseline.py`.** No test file exists for `data/baseline.py` yet, unlike `dataset.py` and `oracle.py`.
- [ ] **Review the fixes Person B made to your files** — `Oracle.get_labels()` (was missing, crashed `StudentModel.train()`) and `StudentModel`'s feature scaling (was missing, caused convergence warnings and an unfair comparison — RL initially looked like it was clearly winning, which turned out to be partly a scaling artifact). Make sure both match what you intended.
- [ ] **Second dataset:** `project-context.md` Section 8 names Adult Census Income as the flagship dataset for Phase 2 — nobody's touched it yet.
- [ ] Help connect the real baseline loops (`run_random_sampling`/`run_uncertainty_sampling`) to Person C's SageMaker pipeline — currently only run locally.

---

## Person B — RL Environment & Agent

*(Full detail in `handoffs/person-b-phase1.md`.)*

- [ ] **Reward/observation tuning.** Current result: RL is statistically tied with uncertainty sampling on accuracy (5-seed mean), with lower variance but no clear edge. The 3-element observation vector (pool uncertainty, labels-used fraction, class balance) was never iterated on — worth trying a richer state.
- [ ] **Add `tests/test_evaluate.py` and `tests/test_run_experiment.py`.**
- [ ] Wire `agent/train.py`/`agent/evaluate.py` into Person C's SageMaker pipeline.
- [ ] Run the same comparison on Adult Census Income once Person A's split is ready.

---

## Person C — AWS Infra + Eval

*(Full detail in `handoffs/person-c-phase1.md`'s own "What the Next Person Should Do" section — copied here since A and B need to act on parts of it.)*

- [ ] **Connect the real Random/Uncertainty/RL loops to the SageMaker pipeline.** `infra/launch_training_job.py` and `infra/training/train.py` are built and tested end-to-end, but per your own handoff note, the current SageMaker classifier is only an infrastructure smoke test — it's never run the team's actual `data/baseline.py` or `env/`/`agent/` logic.
- [ ] **Retire or clearly relabel the old synthetic example CSVs** (`local-test/results/random-seed42.csv`, `local-test/results/uncertainty-seed42.csv`). Real experiment output now exists in `local-test/results/experiments/` — the old ones were a genuine collision risk (same `run_id` convention, and `load_result_directory()`'s recursive glob merged them with real output once, corrupting the aggregation until the real output was moved to its own subdirectory).
- [ ] **Add infra/eval tests** — none exist yet for the S3/SageMaker scripts or `eval/run_experiment.py`.
- [ ] Optional, Phase 2: migrate SageMaker SDK v2 → v3 (currently produces a deprecation warning).

---

## Cross-cutting / Unassigned

- [ ] **GitHub Actions CI mystery.** At one point, merged PRs (#7, #8) triggered zero workflow runs — no error, no pending state, just silence. Never fully diagnosed (ruled out: disabled workflow, outside-contributor approval gating, missing workflow file on `main`). Whoever has repo admin access should check Settings → Actions → General → "Actions permissions" specifically — that's the one thing that was never checked.
- [ ] **Week 6 handoff / Phase 2 rotation.** `project-context.md` Section 3 calls for a group walkthrough session before rotating roles (A→infra, B→baseline, C→RL agent per Section 4). With Person B's and Person C's handoffs written and a real first comparison result in hand, this is close — worth scheduling once Person A's handoff exists too.
