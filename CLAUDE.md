# CLAUDE.md

## Project
RL-driven active learning on UCI tabular datasets. An RL agent learns which unlabeled samples are most valuable to label (maximizing accuracy gain per label), compared against random sampling and uncertainty sampling baselines. See `project-context.md` for the full plan.

## Tech stack
- **Classifier ("student" model):** Scikit-learn
- **RL environment:** Gymnasium (custom env in `env/`)
- **RL agent:** Stable-Baselines3 (PyTorch under the hood)
- **Cloud:** AWS S3 (storage) + SageMaker (training jobs)
- **Experiment tracking:** Weights & Biases
- **Dataset:** UCI Machine Learning Repository (tabular only)

**Not used:** Hugging Face, Ollama, TensorFlow/Keras — not a fit for this stack; don't suggest them.

## Repo structure
```
data/      # dataset prep, oracle (held-back true labels)
env/       # Gymnasium environment: state/action/reward logic
agent/     # SB3 training code
infra/     # AWS/SageMaker/S3 scripts
eval/      # plotting, comparison against baselines
handoffs/  # per-person handoff docs written before task rotation
```

## Commands
- Install deps: `pip install -r requirements.txt`
- Verify env: `python -c "import sklearn, gymnasium, stable_baselines3, torch, boto3, sagemaker; print('OK')"`
- Launch hello-world SageMaker job: `cd infra && python launch_hello_world.py`

## Conventions
- Reward functions live in `env/`, not inline in agent training scripts.
- Keep datasets and model checkpoints out of git (see `.gitignore`) — raw data and artifacts belong in S3, not the repo.
- Each person works on their own branch (`person-a-...`, `person-b-...`, `person-c-...`); merge to `main` via PR.
- Before rotating onto a new component (Week 6), the previous owner fills out a handoff doc in `handoffs/` using the template in `project-context.md` Section 5 — read that doc before touching their code.

## Gotchas
- `infra/launch_hello_world.py` requires `ROLE_ARN`, `BUCKET`, and `REGION` filled in before it will run — these are placeholders, not real values.
- SageMaker instance type should stay small (`ml.t3.medium` or similar) — this project's UCI datasets are small tabular data, no GPU needed.
- AWS credentials should never be committed — `.aws/`, `credentials`, and `.env` are gitignored; double check before pushing if touching AWS config.

## Current focus
See `project-context.md` Section 8 ("Open Decisions") for what's still unresolved (dataset choice, RL algorithm, reward formula, labeling budget). Update this section as decisions get made.
