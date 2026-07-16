# RL-Driven Active Learning on UCI Data — Project Context

*Use this file as shared context for any teammate (or AI assistant) picking up a piece of this project. Keep it updated as decisions change.*

---

## 1. Project Overview

**Goal:** Build a reinforcement learning agent that learns which unlabeled data points are most valuable to label, instead of using a fixed heuristic (random or uncertainty sampling). We measure success by how much accuracy the agent achieves per label used, compared to baselines.

**Why this matters:** Labeling data is expensive in real ML systems. Active learning reduces that cost. Using RL instead of hand-designed heuristics lets the agent discover its own labeling strategy.

**Core RL framing:**
- **State:** current model's uncertainty across the unlabeled pool, number of labels used so far, class balance
- **Action:** select which unlabeled sample(s) to query next
- **Reward:** accuracy gain after retraining with the new label
- **Episode:** starts with a small labeled seed set, ends at a labeling budget (e.g. 100 labels) or target accuracy

**Baselines we compare against:**
- Random sampling
- Standard uncertainty sampling (classic active learning heuristic)

---

## 2. Tech Stack

| Purpose | Tool |
|---|---|
| Classifier ("student" model) | Scikit-learn |
| RL environment | Gymnasium (custom env) |
| RL agent | Stable-Baselines3 (PyTorch-based under the hood) |
| Cloud training | AWS SageMaker Training Jobs |
| Cloud storage | AWS S3 (dataset, checkpoints, logs) |
| Experiment tracking | Weights & Biases (or SageMaker Experiments) |
| Optional stretch | SageMaker Endpoint + simple demo UI (Streamlit) |

**Explicitly not used:** Hugging Face, Ollama — these are built for language models, not tabular RL problems, and aren't a fit here.

**Dataset source:** UCI Machine Learning Repository (tabular datasets — e.g. Adult Income, Wine Quality, Breast Cancer Wisconsin). UCI datasets are static, fully labeled collections — they must be reframed as an RL problem by treating "revealing a label" as the action and "accuracy improvement" as the reward.

---

## 3. Timeline (14–16 weeks)

**Week 1 — AWS Setup (all together)**
- AWS account + IAM roles for all 3 members
- S3 bucket structure
- Billing alarm (protect against surprise cost)
- "Hello world" SageMaker training job everyone can run

**Weeks 2–5 — Phase 1: Solo ownership (parallel)**
- **Person A:** Data + baseline — UCI dataset prep, oracle setup (held-back true labels), sklearn baseline model, random sampling curve
- **Person B:** RL agent — Gymnasium environment, reward design, first working SB3 agent
- **Person C:** AWS infra + eval scaffolding — S3 read/write pipeline, SageMaker training job wrapper, plotting setup

**Week 6 — Handoff + documentation**
- Each person fills out the Handoff Template (Section 5 below) for their piece
- Group walkthrough session: each person explains their code to the other two before rotating

**Weeks 7–10 — Phase 2: Rotation**
- A → AWS infra (extend S3/SageMaker setup, add experiment tracking)
- B → Baseline (polish, add a second UCI dataset for comparison)
- C → RL agent (B available to mentor, since B has the most context)

**Weeks 11–12 — Integration**
- Combine all three pieces: SageMaker job trains both baseline and RL agent → results to S3 → eval script pulls and plots
- Debug end-to-end run as a team

**Weeks 13–14 — Stretch goals**
- SageMaker Endpoint + small demo UI
- Second dataset run for robustness
- Ablations (different reward designs, different query budgets)

**Weeks 15–16 — Write-up + polish**
- Final report/poster: problem statement, architecture diagram (incl. AWS), results/comparison plots, individual contributions/learnings

---

## 4. Task Division Summary

| Phase | Person A | Person B | Person C |
|---|---|---|---|
| Phase 1 (Wk 2–5) | Data + baseline | RL agent | AWS infra + eval |
| Phase 2 (Wk 7–10) | AWS infra | Baseline (polish) | RL agent |

**Why rotate:** ensures all three members get hands-on experience with the RL core, the AWS/cloud pipeline, and the data/baseline work — rather than each person specializing narrowly in one slice.

---

## 5. Handoff Template

*Each person fills this out for their Phase 1 piece before the Week 6 walkthrough.*

```markdown
### Handoff: [Component Name] — [Your Name]

**What this component does:**
(1-2 sentences)

**Key files:**
- `path/to/file.py` — what it does

**How to run it:**
(exact commands)

**Design decisions & why:**
(e.g. why this reward function, why this dataset, why this hyperparameter)

**Known issues / things I didn't get to:**
(be honest — this saves the next person time)

**Gotchas:**
(anything non-obvious that would trip someone up)

**What I'd do next if I kept working on this:**
(gives the next owner a starting point)
```

---

## 6. Practical Notes

- Keep AWS instance sizes small (`ml.t3.medium` or similar) — UCI datasets are small, no GPU needed
- Set a billing alarm on day one, shut down endpoints/instances when not in use
- Git commit discipline matters more here — clear messages and docstrings make the rotation work
- Keep a shared running doc of decisions and gotchas as you go
- The Week 6 walkthrough is load-bearing — don't skip or rush it

---

## 7. Repo Structure & Git Workflow

**One shared repo, not three separate ones.** Everyone works out of a single GitHub repo from day one — separate repos make the Week 6 handoff and Weeks 11-12 integration much harder to stitch together.

**Folder structure (set up in Week 1, even with placeholder files):**

```
project-repo/
├── data/              # Person A's dataset prep, oracle setup
├── env/               # Person B's Gymnasium environment
├── agent/             # Person B's SB3 training code
├── infra/             # Person C's AWS/SageMaker/S3 scripts
├── eval/              # Person C's plotting/comparison code
├── handoffs/          # Each person's handoff .md before rotating (see Section 5)
├── project-context.md # This file
└── README.md
```

**Git workflow:**
1. `main` branch stays stable/working at all times
2. Each person works on their own branch during Phase 1 (`person-a-baseline`, `person-b-agent`, `person-c-infra`)
3. Merge into `main` via pull request at the Week 6 handoff point — the other two review the code before rotating into it
4. During Phase 2, whoever rotates into a component branches off `main` again (e.g. `person-c-agent-v2`) to build on top of what's there

**Repo admin:** one person owns permissions and protecting `main` — doesn't need to be a fixed role, just needs an owner so it doesn't fall through the cracks.

---

## 8. Open Decisions (fill in as you go)

- [ ] Final UCI dataset(s) chosen:
- [ ] RL algorithm chosen (DQN / PPO / bandit-first?):
- [ ] Labeling budget per episode:
- [ ] Reward function exact formula:
- [ ] Stretch goal(s) confirmed:
