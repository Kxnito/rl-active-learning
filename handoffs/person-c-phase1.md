# Person C Phase 1 — AWS Infrastructure & Evaluation

## Status

✅ Phase 1 Complete

This component provides the AWS infrastructure and evaluation scaffolding for the RL Active Learning project.

The entire infrastructure pipeline has been tested end-to-end:

```
Local dataset
    ↓
Upload to S3
    ↓
Launch SageMaker training job
    ↓
SageMaker downloads dataset
    ↓
Runs train.py
    ↓
Produces model + metrics
    ↓
Uploads artifacts to S3
    ↓
Evaluation downloads results
    ↓
Summary metrics + plots
```

---

## What this component owns

### AWS Infrastructure

- AWS configuration
- S3 upload/download utilities
- SageMaker training job launcher
- SageMaker-compatible training entry point

### Evaluation

- Result validation
- Result loading
- Summary metrics
- Learning curve plotting
- Method comparison

---

## Files

### Infrastructure

```
infra/
```

| File | Purpose |
|------|---------|
| config.py | Loads AWS configuration from `.env` |
| s3_utils.py | Reusable S3 upload/download/list functions |
| upload_data.py | Upload processed datasets to S3 |
| download_results.py | Download experiment results from S3 |
| launch_training_job.py | Launch SageMaker training jobs |
| training/train.py | SageMaker training entry point |
| training/requirements.txt | Packages installed inside SageMaker |

---

### Evaluation

```
eval/
```

| File | Purpose |
|------|---------|
| load_results.py | Loads & validates experiment CSVs |
| metrics.py | Computes AUC, reward, summary metrics |
| compare_methods.py | Produces run_summary.csv & method_summary.csv |
| plot_learning_curves.py | Generates learning curve plots |

---

## Environment setup

```
python3.12 -m venv venv

source venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
```

Required environment variables

```
AWS_REGION

S3_BUCKET

SAGEMAKER_ROLE_ARN

PROJECT_PREFIX
```

---

## Commands

### Verify AWS

```bash
aws sts get-caller-identity

aws configure get region

aws s3 ls s3://<bucket>/
```

---

### Upload dataset

```bash
python -m infra.upload_data \
  --file data/processed/breast_cancer.csv \
  --dataset breast-cancer \
  --stage processed
```

---

### Local training

```bash
SM_CHANNEL_TRAINING=local-test/training \
SM_MODEL_DIR=local-test/model \
SM_OUTPUT_DATA_DIR=local-test/output \
python infra/training/train.py \
  --target-column target
```

Expected outputs

```
model.joblib

metrics.json
```

---

### Launch SageMaker training

```bash
python -m infra.launch_training_job \
  --input-s3-uri \
  s3://<bucket>/rl-active-learning/datasets/breast-cancer/processed/ \
  --target-column target \
  --instance-type ml.m5.large \
  --wait
```

Expected output

```
training-output/

model.tar.gz

output.tar.gz
```

---

### Download experiment results

```bash
python -m infra.download_results \
  --prefix results/ \
  --output-dir outputs/downloaded/results
```

---

### Compare methods

```bash
python -m eval.compare_methods \
  --input-dir outputs/downloaded/results
```

Outputs

```
outputs/evaluation/

run_summary.csv

method_summary.csv
```

---

### Generate plots

```bash
python -m eval.plot_learning_curves \
  --input-dir outputs/downloaded/results \
  --dataset breast-cancer \
  --output outputs/plots/learning_curves.png
```

---

## Result CSV format

Every experiment should output

```
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

One row = one point on one learning curve.

---

## Design decisions

- AWS configuration stored in `.env`
- One shared S3 bucket using project prefixes
- SageMaker SDK v2 (SKLearn estimator)
- Logistic Regression is an infrastructure smoke test only
- Common CSV schema shared across Random, Uncertainty, and RL methods
- Learning curve AUC included to measure label efficiency

---

## Known issues

- SageMaker SDK v2 shows a deprecation warning.
- Logistic Regression may show a convergence warning without feature scaling.
- Person A's dataset loader has not yet been integrated.
- Person B's RL training code has not yet been integrated.
- Current Random/Uncertainty/RL comparison CSVs are synthetic testing data only.

---

## AWS permissions required

The SageMaker execution role must have

```
s3:ListBucket

s3:GetObject

s3:PutObject
```

for the project bucket.

---

## What the next person should do

1. Integrate Person A's finished dataset pipeline.
2. Replace the Logistic Regression smoke test with the actual baseline/RL training.
3. Make Random, Uncertainty, and RL all output the shared CSV schema.
4. Run multiple seeds.
5. Generate comparison plots using real experiment data.
6. Add automated tests.
7. (Optional) Upgrade to SageMaker SDK v3 during Phase 2.

---

## Validation completed

Successfully tested

- ✅ S3 upload
- ✅ S3 download
- ✅ SageMaker training job
- ✅ Cloud model artifact generation
- ✅ Cloud metrics generation
- ✅ Result download from S3
- ✅ Result validation
- ✅ Summary metrics
- ✅ Learning curve generation

**Person C Phase 1 is complete.**