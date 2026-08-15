# Person C Phase 1 — AWS Infrastructure & Evaluation

## Status

✅ **Phase 1 Complete**

This component provides the AWS infrastructure and evaluation scaffolding for the RL Active Learning project.

The infrastructure pipeline has been tested end-to-end using Person A's Breast Cancer Wisconsin dataset loader:

```text
data/dataset.py
    ↓
DatasetSplits
    ↓
breast_cancer_splits.npz
    ↓
Upload to Amazon S3
    ↓
Launch SageMaker training job
    ↓
SageMaker downloads dataset
    ↓
Runs infra/training/train.py
    ↓
Produces model + metrics
    ↓
Uploads artifacts to S3
    ↓
Download results
    ↓
Evaluation metrics + comparison plots
```

---

## What This Component Owns

### AWS Infrastructure

* Central AWS configuration
* S3 upload/download utilities
* Dataset serialization and upload
* SageMaker training-job launcher
* SageMaker-compatible training entry point
* Training artifact retrieval

### Evaluation

* Experiment-result validation
* Result loading
* Per-run summary metrics
* Cross-method summary metrics
* Learning-curve AUC
* Multi-seed aggregation support
* Learning-curve plotting
* Random/Uncertainty/RL method comparison

---

## Files

### Infrastructure

```text
infra/
├── config.py
├── s3_utils.py
├── upload_data.py
├── download_results.py
├── launch_training_job.py
└── training/
    ├── train.py
    └── requirements.txt
```

| File                        | Purpose                                                                                  |
| --------------------------- | ---------------------------------------------------------------------------------------- |
| `config.py`                 | Loads AWS configuration from environment variables                                       |
| `s3_utils.py`               | Reusable S3 upload, download, listing, and existence-check operations                    |
| `upload_data.py`            | Calls Person A's dataset loader, serializes the splits to `.npz`, and uploads them to S3 |
| `download_results.py`       | Downloads SageMaker outputs or experiment CSV results from S3                            |
| `launch_training_job.py`    | Configures and launches SageMaker training jobs                                          |
| `training/train.py`         | SageMaker-compatible training entry point and infrastructure smoke test                  |
| `training/requirements.txt` | Python dependencies installed inside the SageMaker training container                    |

### Evaluation

```text
eval/
├── load_results.py
├── metrics.py
├── compare_methods.py
└── plot_learning_curves.py
```

| File                      | Purpose                                                                              |
| ------------------------- | ------------------------------------------------------------------------------------ |
| `load_results.py`         | Loads and validates standardized experiment CSV files                                |
| `metrics.py`              | Computes learning-curve AUC, normalized AUC, reward totals, and run/method summaries |
| `compare_methods.py`      | Generates `run_summary.csv` and `method_summary.csv`                                 |
| `plot_learning_curves.py` | Generates validation-accuracy-vs-label-budget plots                                  |

---

## Dataset Integration

Person A's dataset loader is located at:

```text
data/dataset.py
```

It loads the Breast Cancer Wisconsin dataset and creates four separate data partitions:

* **Seed set** — initially labeled examples available to the student model
* **Unlabeled pool** — examples available for active-learning queries
* **Validation set** — used for reward calculation and model evaluation during the episode
* **Test set** — held out until final evaluation

The infrastructure calls:

```python
load_dataset(
    seed_size=10,
    val_size=100,
    test_size=100,
    random_state=42,
)
```

The resulting `DatasetSplits` object is serialized to:

```text
outputs/breast_cancer_splits.npz
```

The `.npz` artifact contains:

```text
seed_X
seed_y
pool_X
pool_y
val_X
val_y
test_X
test_y
```

The currently tested split sizes are:

```text
Seed:       (10, 30)
Pool:       (359, 30)
Validation: (100, 30)
Test:       (100, 30)
```

`pool_y` contains the hidden labels for the unlabeled pool. It should eventually be provided only to the Oracle and must not be directly exposed to the RL agent.

---

## Environment Setup

Create and activate a virtual environment:

```bash
python3.12 -m venv venv
source venv/bin/activate
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

Create the local environment configuration:

```bash
cp .env.example .env
```

Required environment variables:

```text
AWS_REGION
S3_BUCKET
SAGEMAKER_ROLE_ARN
PROJECT_PREFIX
```

Example structure:

```text
AWS_REGION=us-east-2
S3_BUCKET=rl-active-learning-project
PROJECT_PREFIX=rl-active-learning
SAGEMAKER_ROLE_ARN=arn:aws:iam::<account-id>:role/<sagemaker-role>
```

Do not commit `.env` or AWS credentials to Git.

---

## Verify AWS Access

Verify the current AWS identity:

```bash
aws sts get-caller-identity
```

Verify the configured region:

```bash
aws configure get region
```

Verify access to the project bucket:

```bash
aws s3 ls s3://rl-active-learning-project/
```

---

## Upload the Dataset

Run from the repository root:

```bash
python -m infra.upload_data
```

This:

1. Calls `data.dataset.load_dataset()`
2. Creates the seed/pool/validation/test splits
3. Saves them as a compressed NumPy artifact
4. Uploads the artifact to S3

Local artifact:

```text
outputs/breast_cancer_splits.npz
```

Current S3 location:

```text
s3://rl-active-learning-project/rl-active-learning/data/processed/breast_cancer_splits.npz
```

---

## Launch SageMaker Training

Run:

```bash
python -m infra.launch_training_job \
  --input-s3-uri s3://rl-active-learning-project/rl-active-learning/data/processed/breast_cancer_splits.npz \
  --wait
```

Optional arguments include:

```text
--random-state
--instance-type
--wait
```

The current default training instance is:

```text
ml.m5.large
```

The SageMaker training job receives the `.npz` file through the `training` input channel.

Inside SageMaker, the dataset is available under:

```text
/opt/ml/input/data/training/
```

The training script reads the directory through:

```text
SM_CHANNEL_TRAINING
```

Model artifacts are written under:

```text
/opt/ml/model/
```

Metrics and other output data are written under:

```text
/opt/ml/output/data/
```

---

## Current Training Smoke Test

`infra/training/train.py` currently trains a Scikit-learn Logistic Regression classifier.

This is an **infrastructure smoke test only**. It is not the final active-learning baseline or RL algorithm.

The smoke test:

1. Loads Person A's prepared `.npz` artifact
2. Trains Logistic Regression using `seed_X` and `seed_y`
3. Evaluates the model using `val_X` and `val_y`
4. Saves the trained model
5. Saves validation metrics

The successfully tested SageMaker run produced:

```text
validation_accuracy=0.860000
```

and:

```json
{
  "algorithm": "logistic-regression-smoke-test",
  "validation_accuracy": 0.86,
  "seed_rows": 10,
  "pool_rows": 359,
  "validation_rows": 100,
  "test_rows": 100,
  "random_state": 42
}
```

Expected training artifacts:

```text
model.joblib
metrics.json
```

SageMaker packages these into:

```text
model.tar.gz
output.tar.gz
```

---

## Download SageMaker Results

For a completed SageMaker job:

```bash
python -m infra.download_results \
  --job-name <training-job-name>
```

Example:

```bash
python -m infra.download_results \
  --job-name active-learning-smoke-20260812-145216
```

The downloaded files are placed under:

```text
outputs/downloaded/<training-job-name>/
```

Example:

```text
outputs/downloaded/
└── active-learning-smoke-20260812-145216/
    ├── output.tar.gz
    └── metrics.json
```

`download_results.py` also retains support for downloading experiment CSV files from a results prefix for later baseline/RL integration.

---

## Experiment Result CSV Contract

Random sampling, uncertainty sampling, and the RL agent should all produce the same result format.

Required columns:

```text
run_id
method
dataset
seed
step
labels_used
val_accuracy
test_accuracy
reward
```

One row represents one point on one active-learning learning curve.

Example:

```csv
run_id,method,dataset,seed,step,labels_used,val_accuracy,test_accuracy,reward
random-42,random,breast_cancer,42,0,10,0.86,,0.00
random-42,random,breast_cancer,42,1,20,0.88,,0.02
random-42,random,breast_cancer,42,2,30,0.90,,0.02
random-42,random,breast_cancer,42,3,40,0.91,,0.01
random-42,random,breast_cancer,42,4,50,0.92,0.93,0.01
```

### Important Test-Set Rule

`test_accuracy` should normally be blank during intermediate active-learning steps.

The validation set is used during the episode to measure performance and calculate reward.

The test set is held out for final reporting and should not influence:

* Query selection
* Model training decisions
* Reward calculation
* Hyperparameter tuning

Therefore, `test_accuracy` should normally only be populated during the final test evaluation.

---

## Compare Methods

Run:

```bash
python -m eval.compare_methods \
  --input-dir <results-directory>
```

For local synthetic testing:

```bash
python -m eval.compare_methods \
  --input-dir local-test/results
```

Outputs:

```text
outputs/evaluation/
├── run_summary.csv
└── method_summary.csv
```

The evaluation layer currently calculates:

* Initial validation accuracy
* Final validation accuracy
* Final test accuracy
* Total reward
* Learning-curve AUC
* Normalized learning-curve AUC
* Mean metrics across runs/seeds
* Standard deviation across runs/seeds

Standard deviation is `NaN` when only one run exists for a method. This is expected. Multiple seeds are required to estimate run-to-run variability.

---

## Generate Learning-Curve Plots

Run:

```bash
python -m eval.plot_learning_curves \
  --input-dir <results-directory> \
  --dataset breast_cancer
```

For the local synthetic test:

```bash
python -m eval.plot_learning_curves \
  --input-dir local-test/results \
  --dataset breast_cancer
```

Default output:

```text
outputs/plots/learning_curves.png
```

The plot shows:

```text
Validation Accuracy
        vs.
Labels Used
```

and aggregates results by method and labeling budget.

When multiple seeds are available, the plot displays the mean validation accuracy and variability across runs.

Datasets should be evaluated separately rather than combining learning curves from different datasets.

---

## Evaluation Validation Performed

The evaluation pipeline was tested using synthetic Random and Uncertainty result CSVs.

The synthetic comparison successfully produced method-level results such as:

```text
Method          Final Val Accuracy    Normalized AUC
uncertainty            0.94               0.9075
random                 0.92               0.8950
```

These values are **synthetic infrastructure/evaluation test data only** and must not be reported as project experimental findings.

The purpose of this test was to verify that:

* Multiple methods can be loaded
* Result schemas are validated
* Runs can be summarized
* Methods can be compared
* Learning-curve AUC is calculated
* Learning curves can be plotted

---

## Design Decisions

### Common Result Schema

Random sampling, uncertainty sampling, and RL use the same CSV contract.

This keeps evaluation independent of how each strategy is implemented internally.

### Person A Owns Dataset Splitting

Dataset splitting remains in `data/dataset.py`.

The infrastructure consumes Person A's `DatasetSplits` rather than duplicating train/validation/test splitting logic.

### NumPy `.npz` Dataset Artifact

The prepared dataset is stored as:

```text
breast_cancer_splits.npz
```

instead of CSV because the active-learning dataset consists of eight related NumPy arrays.

### Validation/Test Separation

Validation accuracy is used for learning-curve measurement and reward.

The test set remains isolated until final evaluation to prevent leakage.

### Logistic Regression Smoke Test

Logistic Regression currently exists only to verify that:

```text
S3 → SageMaker → training → metrics/model → S3
```

works successfully.

It is not intended to replace Person A's baseline or Person B's RL implementation.

### Learning-Curve AUC

Final accuracy alone does not fully measure active-learning efficiency.

Normalized learning-curve AUC captures how well a method performs across the entire labeling budget.

### Multiple Seeds

The evaluation layer supports multiple random seeds and computes mean and standard deviation across runs.

Real experiments should use multiple seeds before drawing conclusions about method performance.

---

## AWS/SageMaker Notes

### SageMaker Debugger

The initial SageMaker training-job submission failed because SageMaker Debugger is unavailable to new customers.

The launcher therefore explicitly disables Debugger/profiling for the current training job configuration.

### SageMaker SDK Version

The current implementation uses SageMaker Python SDK v2 and the `SKLearn` estimator.

SDK v2 currently produces a deprecation warning because AWS recommends SageMaker SDK v3 for new development.

The existing v2 implementation is working and was retained for Phase 1 to avoid an unnecessary infrastructure migration during the initial implementation.

Migrating to SDK v3 can be considered during Phase 2.

### Instance Type

The smoke test currently uses:

```text
ml.m5.large
```

No GPU is required for the current UCI/Scikit-learn workload.

---

## AWS Permissions Required

The SageMaker execution role needs access to the project S3 bucket, including the equivalent of:

```text
s3:ListBucket
s3:GetObject
s3:PutObject
```

IAM permissions should follow least-privilege principles and be limited to the project resources where practical.

---

## Git / Security Notes

Do not commit:

```text
.env
AWS credentials
__pycache__/
*.pyc
```

Generated outputs should also generally remain outside source control unless intentionally retained as project artifacts.

Suggested `.gitignore` entries include:

```gitignore
.env
.venv/
venv/
__pycache__/
*.py[cod]
```

---

## Known Issues / Remaining Integration

* SageMaker Python SDK v2 produces a deprecation warning.
* The current SageMaker classifier is only an infrastructure smoke test.
* Person A's dataset loader **is integrated**, but the full Random/Uncertainty experiment loop still needs to be connected to the cloud pipeline.
* Person B's RL environment and agent still need to be connected to the same training/result pipeline.
* Current Random/Uncertainty evaluation CSVs are synthetic test data.
* Real experiments still need multiple seeds before standard-deviation estimates and method comparisons are meaningful.
* Automated integration tests can be expanded during Phase 2.

---

## What the Next Person Should Do

1. Connect Person A's Random and Uncertainty baseline experiment loops to the existing SageMaker/result pipeline.
2. Connect Person B's RL environment and agent to the same pipeline.
3. Make Random, Uncertainty, and RL emit the shared result CSV schema.
4. Run real experiments using multiple random seeds.
5. Replace synthetic evaluation CSVs with real experiment outputs.
6. Generate final accuracy-vs-label-budget comparison plots.
7. Add or expand automated infrastructure and result-schema tests.
8. Optionally migrate SageMaker SDK v2 to v3 during Phase 2.

---

## Validation Completed

Successfully tested:

* ✅ Person A dataset loader integration
* ✅ Seed/pool/validation/test serialization
* ✅ `.npz` dataset generation
* ✅ S3 dataset upload
* ✅ S3 download
* ✅ SageMaker training-job creation
* ✅ SageMaker input-channel download
* ✅ Dataset loading inside SageMaker
* ✅ Cloud model training
* ✅ Cloud model artifact generation
* ✅ Cloud metrics generation
* ✅ SageMaker result retrieval
* ✅ Result CSV validation
* ✅ Per-run summary metrics
* ✅ Cross-method comparison
* ✅ Learning-curve AUC
* ✅ Learning-curve plot generation
* ✅ Synthetic Random-vs-Uncertainty evaluation test

## Phase 1 Result

The AWS infrastructure and evaluation scaffolding required for Person C Phase 1 are operational and tested.

The next integration stage is to connect the team's real Random, Uncertainty, and RL experiment implementations to the existing infrastructure and common evaluation contract.

**Person C Phase 1 is complete.**
