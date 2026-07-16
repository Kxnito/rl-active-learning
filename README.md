# RL Active Learning on UCI Data

Reinforcement learning agent that learns *which* unlabeled data points are most valuable to label — instead of using a fixed heuristic like random or uncertainty sampling. We measure success by how much model accuracy the agent achieves per label used, compared against standard baselines.

## Why

Labeling data is expensive in real-world ML. Active learning reduces that cost by being selective about what gets labeled. Most active learning uses hand-designed heuristics; this project trains an RL agent to learn its own labeling strategy instead, optimizing directly for accuracy gained per label.

## How it works

- **State:** current model's uncertainty across the unlabeled pool, labels used so far, class balance
- **Action:** select which unlabeled sample(s) to query next
- **Reward:** accuracy improvement after retraining with the new label
- **Episode:** starts with a small labeled seed set, ends at a labeling budget or target accuracy

Compared against two baselines: random sampling and standard uncertainty sampling.

## Tech stack

- **Scikit-learn** — the classifier being actively trained
- **Gymnasium** — custom active-learning environment
- **Stable-Baselines3** — RL agent (PyTorch under the hood)
- **AWS S3 + SageMaker** — cloud storage and training
- **UCI Machine Learning Repository** — dataset source

## Repo structure

```
data/              # dataset prep, oracle setup
env/               # Gymnasium environment
agent/             # RL agent training code
infra/             # AWS / SageMaker / S3 scripts
eval/              # evaluation, plotting, comparisons
handoffs/          # handoff docs written before task rotation
```

## Getting started

See [`SETUP.md`](./SETUP.md) for Day 1 setup (local environment, AWS, hello-world SageMaker test).

See [`project-context.md`](./project-context.md) for the full project plan: timeline, task rotation, tech decisions, and open items.

## Team

| Phase | Person A | Person B | Person C |
|---|---|---|---|
| Phase 1 | Data + baseline | RL agent | AWS infra + eval |
| Phase 2 | AWS infra | Baseline (polish) | RL agent |

## Status

🚧 In progress — see `project-context.md` Section 8 for open decisions.
