# AGENTS.md

## Project
RL-driven active learning on UCI tabular datasets. An RL agent learns which unlabeled samples are most valuable to label (maximizing accuracy gain per label), compared against random sampling and uncertainty sampling baselines. See `project-context.md` for the full plan.

## Tech stack
- **Classifier ("student" model):** Scikit-learn
- **RL environment:** Gymnasium (custom env in `env/`)
- **RL agent:** Stable-Baselines3 (PyTorch under the hood) — must use `MaskablePPO` from `sb3-contrib`, not plain SB3 DQN/PPO, since the unlabeled pool shrinks every step (needs action masking)
- **Cloud:** AWS S3 (storage) + SageMaker (training jobs)
- **Experiment tracking:** Weights & Biases
- **Dataset:** UCI Machine Learning Repository (tabular only)

**Not used:** Hugging Face, Ollama, TensorFlow/Keras — not a fit for this stack; don't suggest them.

## Repo structure
```
data/      # dataset prep, oracle (held-back true labels) — dataset.py, oracle.py, baseline.py
env/       # Gymnasium environment: state/action/reward logic — active_learning_env.py
agent/     # SB3 training code — train.py
infra/     # AWS/SageMaker/S3 scripts
eval/      # plotting, comparison against baselines
handoffs/  # per-person handoff docs written before task rotation
```

## Commands
- Install deps: `pip install -r requirements.txt`
- Verify env: `python -c "import sklearn, gymnasium, stable_baselines3, sb3_contrib, torch, boto3, sagemaker; print('OK')"`
- Launch hello-world SageMaker job: `cd infra && python launch_hello_world.py`

## Conventions
- Reward functions live in `env/`, not inline in agent training scripts.
- Keep datasets and model checkpoints out of git (see `.gitignore`) — raw data and artifacts belong in S3, not the repo.
- Each person works on their own branch (`person-a-...`, `person-b-...`, `person-c-...`); merge to `main` via PR.
- **`main` is protected by a GitHub ruleset** — direct pushes are rejected, even for small doc changes. Always: `git checkout -b <branch-name>`, push the branch, open a PR, get 1 approval, then merge.
- Before rotating onto a new component (Week 6), the previous owner fills out a handoff doc in `handoffs/` using the template in `project-context.md` Section 5 — read that doc before touching their code.

## AWS account structure
- One shared AWS account for the team (not one per person) — full setup steps in `AWS_SETUP.md`
- Root is used only for account creation + billing alarm, never for daily work
- Each teammate has their own regular-worker IAM user (`AmazonSageMakerFullAccess` + `AmazonS3FullAccess`) for day-to-day use — use these, not admin, not root, when running scripts
- An `admin-<name>` IAM user exists separately for rare account-level changes only

## Gotchas
- `infra/launch_hello_world.py` requires `ROLE_ARN`, `BUCKET`, and `REGION` filled in before it will run — these are placeholders, not real values.
- SageMaker instance type should stay small (`ml.t3.medium` or similar) — this project's UCI datasets are small tabular data, no GPU needed.
- AWS credentials should never be committed — `.aws/`, `credentials`, and `.env` are gitignored; double check before pushing if touching AWS config.
- If `git push` is rejected with "GH013: Repository rule violations," that means you tried to push directly to `main` — branch and PR instead, don't try to bypass it.
- The unlabeled pool shrinks by one every step as labels get revealed — plain SB3 DQN/PPO assume a fixed action space and don't support this. Use `MaskablePPO` (`sb3-contrib`) with `env.action_masks()`, not vanilla DQN.

## Current focus
See `project-context.md` Section 8 ("Open Decisions") for what's still unresolved (dataset choice, RL algorithm, reward formula, labeling budget). Update this section as decisions get made.

## Keeping this file current
Claude Code is authorized to edit this file directly when it learns something important — a decision, a gotcha, a resolved item in `project-context.md` Section 8 — rather than only suggesting the edit. Keep additions concise and dated in spirit (most-recent-relevant, not a changelog); if something here turns out to be wrong or superseded, fix it in place instead of leaving both versions.