# AWS Account Structure & Setup

Referenced from `AGENTS.md`. One shared AWS account for the team — not one account per person.

## Account tiers

- **Root** — used only for account creation and setting the billing alarm. Never used for daily work; don't generate access keys for root.
- **Worker IAM users (one per teammate)** — `AmazonSageMakerFullAccess` + `AmazonS3FullAccess`. This is what everyone uses day-to-day: running training jobs, reading/writing S3, launching `infra/launch_hello_world.py`.
- **`admin-<name>` IAM users** — separate from each teammate's worker user, used only for rare account-level changes (IAM policy edits, billing config, new resource types). Not used for routine scripts or training jobs.

## Setup steps (whole group, together, Week 1)

1. Create the AWS account (check for AWS Educate / student credits first).
2. From root, set a **Billing Alarm** in AWS Budgets before anything else touches SageMaker.
3. Create an S3 bucket for the project (datasets, checkpoints, logs, results).
4. Create a SageMaker execution role (IAM → Roles → SageMaker → `AmazonSageMakerFullAccess` managed policy for now; scope down later if needed).
5. For each teammate, create:
   - a worker IAM user (`AmazonSageMakerFullAccess` + `AmazonS3FullAccess`)
   - an `admin-<name>` IAM user, kept separate, used only for account-level changes
6. Each teammate runs `aws configure` locally with their **worker** IAM user credentials — not admin, not root.

## Gotchas

- Never commit AWS credentials — `.aws/`, `credentials`, and `.env` are gitignored; double-check before pushing if you touched AWS config.
- If you find yourself using your `admin-<name>` credentials for a training job or S3 read/write, stop — that's a sign you should be using your worker user instead.
