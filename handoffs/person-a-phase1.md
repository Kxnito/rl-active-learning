# Person A Phase 1 — Data, Baselines & Oracle

## Status

✅ **Phase 1 Complete**

This phase provides the core data and evaluation infrastructure used by the active-learning pipeline.

It:

* Loads and prepares the Breast Cancer Wisconsin dataset.
* Creates reproducible **seed / pool / validation / test** splits.
* Provides an `Oracle` that simulates requesting labels from the unlabeled pool.
* Prevents the same pool sample from being revealed more than once.
* Implements **random sampling** and **uncertainty sampling** baselines.
* Produces accuracy-vs-label learning curves that can be compared against the RL active-learning agent.

These files are intentionally independent of the RL agent. They provide the common dataset, labeling behavior, and baseline methods that the RL approach should be compared against.

---

## What This Component Owns

### Dataset — `data/dataset.py`

Responsible for loading and splitting the dataset used throughout the project.

Main responsibilities:

* Load the Breast Cancer Wisconsin dataset.
* Separate features and labels.
* Create the initial labeled **seed set**.
* Create the **unlabeled pool** used for active-learning queries.
* Create held-out **validation** and **test** sets.
* Keep splits reproducible through a random seed.
* Return the splits in a format shared by the baseline, Oracle, environment, and agent code.

Conceptually, the resulting data looks like:

```text
Full Dataset
│
├── Seed Set
│   └── Initially labeled training examples
│
├── Pool
│   └── Examples available for active-learning queries
│
├── Validation Set
│   └── Used to measure learning progress / reward
│
└── Test Set
    └── Final held-out evaluation
```

The important distinction is that `pool_X` is available to the learner, while `pool_y` should be treated as hidden and accessed through the Oracle.

---

### Baselines — `data/baseline.py`

Implements the non-RL active-learning methods used as comparison points.

The two baseline strategies are:

**Random sampling**

Selects an unlabeled sample randomly from the remaining pool.

This provides the basic lower-bar comparison:

> Does active selection perform better than simply labeling random examples?

**Uncertainty sampling**

Uses the current student model's predicted probabilities and selects the example the model is least confident about.

For binary classification, this generally means selecting samples whose predicted class probabilities are closest to the decision boundary.

The baseline module also owns the student-model training/evaluation loop needed to produce learning curves.

At each query step, the general process is:

```text
Train student model
        ↓
Evaluate accuracy
        ↓
Choose pool sample
        ↓
Ask Oracle for label
        ↓
Add sample to labeled training set
        ↓
Retrain student model
        ↓
Repeat until budget is exhausted
```

The resulting curve records model performance as more labels become available.

This gives the RL agent something directly comparable against.

---

### Oracle — `data/oracle.py`

Simulates the human annotator in an active-learning system.

The learner should not directly access the hidden pool labels during querying. Instead, it selects a pool index and asks the Oracle to reveal that sample's label.

Conceptually:

```python
label = oracle.query(pool_index)
```

The Oracle:

* Stores/accesses the hidden pool labels.
* Reveals the label corresponding to a requested pool index.
* Tracks which samples have already been queried.
* Tracks the total number of labels revealed.
* Rejects attempts to reveal the same sample twice.

The single-reveal rule is important because an active-learning budget represents the number of unique annotations purchased.

Without this protection, a strategy could accidentally query the same point multiple times and corrupt its label-budget accounting.

---

## Key Files

| File               | Purpose                                                                                                     |
| ------------------ | ----------------------------------------------------------------------------------------------------------- |
| `data/dataset.py`  | Loads Breast Cancer Wisconsin and creates reproducible seed/pool/validation/test splits                     |
| `data/baseline.py` | Implements the student model, random sampling, uncertainty sampling, and accuracy-per-label learning curves |
| `data/oracle.py`   | Simulates label acquisition from the unlabeled pool and enforces one reveal per sample                      |

---

## How the Three Files Work Together

The intended flow is:

```text
data/dataset.py
      │
      │ creates dataset splits
      ▼
 seed / pool / val / test
      │
      ├──────────────────────┐
      │                      │
      ▼                      ▼
data/baseline.py       data/oracle.py
      │                      │
      │ selects index        │ owns hidden labels
      └──────────┬───────────┘
                 │
                 ▼
           Oracle.query()
                 │
                 ▼
          revealed label
                 │
                 ▼
       student model retrains
                 │
                 ▼
       validation/test score
```

The same basic interaction is later reused by the RL environment:

```text
Dataset
   ↓
Active Learning Method
   ├── Random
   ├── Uncertainty
   └── RL Agent
   ↓
Oracle
   ↓
New labeled sample
   ↓
Student Model
   ↓
Evaluation
```

This separation is intentional: the **query strategy changes**, but the dataset and label-revealing mechanism remain consistent.

---

## How to Run It

### Load the dataset

```python
from data.dataset import load_dataset

splits = load_dataset(
    seed_size=20,
    val_size=100,
    test_size=100,
)
```

This returns the dataset partitions needed by the rest of the active-learning pipeline.

---

### Run the tests

```bash
python -m pytest tests/ -W ignore::RuntimeWarning
```

Run this after modifying the dataset, Oracle, baseline student model, or anything that depends on their interfaces.

---

### Run the full experiment

```bash
python -m eval.run_experiment --seed 42
```

This runs the baseline methods and RL agent using the same dataset seed and labeling budget so their learning curves can be compared fairly.

For multiple seeds:

```bash
for seed in 42 43 44 45 46; do
    python -m eval.run_experiment --seed $seed
done
```

---

## Design Decisions & Why

### Fixed seed / pool / validation / test splits

All methods must operate on the same underlying split for a given random seed.

Otherwise, differences between random sampling, uncertainty sampling, and RL could come from different train/test examples rather than from the query strategy itself.

The random seed therefore matters for reproducibility and fair comparison.

---

### Pool features are visible; pool labels are hidden

The learner is allowed to inspect `pool_X`.

That is necessary for active learning because the strategy needs information about the unlabeled examples in order to decide which one to query.

`pool_y`, however, represents labels that have not yet been purchased/revealed.

Those labels should only become available through the Oracle.

This means using `pool_X` for preprocessing is allowed and is **not label leakage**.

Using `pool_y` to influence query selection would be leakage.

---

### Oracle enforces one reveal per sample

Each queried pool sample represents one annotation.

Once a sample has been revealed, querying it again should not consume another meaningful active-learning step.

The Oracle therefore tracks previously revealed indices and prevents duplicate reveals.

Any new querying strategy should preserve this assumption.

---

### Baselines and RL must use the same label accounting

The learning curves use total labeled examples rather than only counting newly queried pool samples.

In other words:

```text
labels_used = initial seed labels + revealed pool labels
```

If the seed contains 20 examples, the first newly queried point brings the total labeled count to 21.

This convention must remain consistent across:

* random sampling
* uncertainty sampling
* RL evaluation

Otherwise, the learning curves will have misaligned x-axes and comparisons will be misleading.

---

### Feature scaling uses the available feature pool

The student model uses standardized features.

An earlier implementation fit `StandardScaler` using only the small initial seed set. With roughly 20 samples, some Breast Cancer Wisconsin features had extremely small variance, causing samples elsewhere in the pool to become very large after scaling.

That produced unstable numerical behavior during model training.

The scaler was changed to fit on the available feature data (`seed_X + pool_X`) rather than repeatedly fitting on only the seed set.

This does not expose hidden labels because it uses only `X`, not `y`.

If another student-model implementation is added later, it should be careful not to reintroduce the seed-only scaling problem.

---

## Expected Baseline Behavior

The baselines serve two different purposes.

### Random sampling

Random sampling is the control.

Results will vary between seeds because the queried samples are selected randomly.

It answers:

> How quickly would the model improve if we made no intelligent labeling decisions?

### Uncertainty sampling

Uncertainty sampling is the stronger traditional active-learning baseline.

It answers:

> Can the learner improve faster by labeling examples it currently finds difficult?

Any RL approach should therefore be compared against **both**.

Beating random alone is not especially meaningful if uncertainty sampling achieves the same or better performance with a much simpler strategy.

---

## Important Invariants

If you modify these files, preserve the following unless the experiment design intentionally changes:

1. `seed_y` is available immediately.
2. `pool_X` can be inspected by the learner.
3. `pool_y` is treated as hidden.
4. Pool labels are obtained through the Oracle.
5. A pool sample can only be revealed once.
6. Validation data is never added to the training set.
7. Test data is never added to the training set.
8. All methods use the same split for a given experiment seed.
9. Label budgets are counted consistently across methods.
10. Random, uncertainty, and RL learning curves use compatible output conventions.

Breaking one of these can make the comparison between methods invalid even if the code still runs.

---

## Gotchas

### Do not use `pool_y` when implementing a query strategy

A query strategy may inspect features and current model predictions, but it should not inspect the true pool labels.

For example, this is invalid:

```python
best_index = choose_using_true_labels(pool_X, pool_y)
```

The entire point of the Oracle abstraction is that those labels are unknown until queried.

---

### Do not accidentally query the same sample twice

Any strategy that keeps its own pool indices must stay synchronized with the Oracle's reveal state.

The Oracle intentionally rejects duplicate reveals.

If a new strategy starts throwing duplicate-query errors, check its index bookkeeping before changing the Oracle.

---

### Keep preprocessing consistent

The baseline student model and RL environment should see features under equivalent preprocessing assumptions.

If one method gets differently scaled data, the resulting comparison is no longer purely about query strategy.

---

### Do not evaluate progress on the test set

Use the validation set for intermediate model evaluation and active-learning decisions.

The test set should remain held out for final evaluation.

Repeatedly evaluating or tuning against the test set turns it into another validation set and makes the reported final accuracy less meaningful.

---

### Small changes to the dataset split affect every downstream result

`dataset.py` sits at the bottom of the entire experiment stack.

Changing:

* split ordering
* random-state behavior
* seed size
* validation size
* test size
* preprocessing

can change baseline and RL results simultaneously.

If results suddenly move after a dataset change, do not immediately assume the agent or baseline implementation broke.

---

## Known Limitations

* Currently centered on the **Breast Cancer Wisconsin** dataset.
* The current student-model/preprocessing setup assumes continuous numerical features.
* Adult Census Income will require additional thought because it contains mixed categorical/numerical features and has different class-balance and dataset-size characteristics.
* Uncertainty sampling is only one traditional active-learning baseline; other strategies such as entropy sampling, margin sampling, query-by-committee, or diversity-aware sampling are not currently implemented.
* Current behavior and hyperparameters were developed around Breast Cancer Wisconsin and should not automatically be assumed to transfer cleanly to other datasets.

---

## What I'd Do Next

1. Add/strengthen unit tests specifically for `dataset.py`, `baseline.py`, and `oracle.py`.
2. Verify deterministic dataset splits for identical random seeds.
3. Add explicit tests proving the Oracle rejects duplicate reveals.
4. Add tests confirming query strategies never access hidden labels.
5. Run baseline sweeps over multiple seeds rather than relying on one run.
6. Add Adult Census Income as the second dataset.
7. Refactor preprocessing if needed so numerical/categorical datasets can share the same active-learning interface.
8. Add additional classical active-learning baselines if stronger comparisons are needed.

---

## TL;DR for the Next Person

If you're trying to understand this part of the repo, start here:

```text
data/dataset.py
```

It defines **what data every method receives**.

Then read:

```text
data/oracle.py
```

It defines **how hidden labels are revealed**.

Then read:

```text
data/baseline.py
```

It shows **how a complete non-RL active-learning loop works**.

Once those three make sense, the RL environment is much easier to understand: the RL agent is essentially replacing the baseline's rule for **which pool sample should be queried next**, while keeping the same overall active-learning problem.