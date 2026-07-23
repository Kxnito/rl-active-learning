# Day 1 Setup Guide

## 1. Local environment (each person)

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Confirm it worked:
```bash
python -c "import sklearn, gymnasium, stable_baselines3, torch, boto3, sagemaker; print('All imports OK')"
```

## 2. AWS setup (whole group, together)

1. Create the AWS account (check for AWS Educate / student credits first)
2. In IAM, create one user per teammate (avoid using root credentials day-to-day)
3. Set up a **Billing Alarm** in AWS Budgets — do this before anything else touches SageMaker
4. Create an S3 bucket, e.g. `your-project-name-data`
5. Create a SageMaker execution role (IAM -> Roles -> search "SageMaker" -> use the managed policy `AmazonSageMakerFullAccess` for now; can be scoped down later)
6. Each teammate runs `aws configure` locally with their own IAM credentials

## 3. Hello-world SageMaker test

Files: `infra/train_hello_world.py` + `infra/launch_hello_world.py`

1. Open `infra/launch_hello_world.py` and fill in:
   - `ROLE_ARN` (from step 2.5 above)
   - `BUCKET` (from step 2.4 above)
   - `REGION`
2. Run it:
   ```bash
   cd infra
   pip install sagemaker boto3
   python launch_hello_world.py
   ```
3. Check the AWS Console → SageMaker → Training Jobs → confirm it shows "Completed"
4. Check the S3 bucket → `hello-world/output/` → confirm `hello_world_result.json` and the placeholder model file landed there

**Have all 3 teammates run this individually** — the goal is confirming everyone's IAM credentials and permissions actually work before Phase 1 begins.

## 4. Repo setup

- Push this folder structure + `project-context.md` to the shared GitHub repo
- Protect `main`, require PRs
- Each person creates their Phase 1 branch: `person-a-baseline`, `person-b-agent`, `person-c-infra`

## 5. Dataset decision

Pick one as a group before ending today (suggested starting point: Breast Cancer Wisconsin or Wine Quality — small, clean, minimal preprocessing).