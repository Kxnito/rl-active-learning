"""
Launches the hello-world training script as a real SageMaker Training Job.

Run this from your local machine (with AWS credentials configured via
`aws configure` or environment variables) to confirm your IAM user can
successfully launch and monitor a SageMaker job.

Before running, fill in:
  - ROLE_ARN: the SageMaker execution role ARN (created in AWS Console ->
    IAM -> Roles -> "AmazonSageMaker-ExecutionRole..." or similar)
  - BUCKET: the S3 bucket created in Week 1 setup
  - REGION: your AWS region (e.g. "us-east-1")

Install dependency first:
    pip install sagemaker boto3
"""

import sagemaker
from sagemaker.sklearn.estimator import SKLearn

# ---- EDIT THESE THREE VALUES ----
ROLE_ARN = "arn:aws:iam::<YOUR_ACCOUNT_ID>:role/<YOUR_SAGEMAKER_EXECUTION_ROLE>"
BUCKET = "<your-project-bucket-name>"
REGION = "us-east-1"
# ----------------------------------

session = sagemaker.Session()

estimator = SKLearn(
    entry_point="train_hello_world.py",
    source_dir=".",                    # folder containing train_hello_world.py
    role=ROLE_ARN,
    instance_type="ml.t3.medium",       # small + cheap, plenty for this test
    instance_count=1,
    framework_version="1.2-1",          # sklearn container version
    py_version="py3",
    sagemaker_session=session,
    output_path=f"s3://{BUCKET}/hello-world/output",
)

# No real training data needed for this first test — SageMaker just needs
# something to mount. Point it at any small file in your bucket, or skip
# the "inputs" argument entirely for a pure smoke test.
estimator.fit(job_name="hello-world-test")

print("Job submitted! Check status in the SageMaker console under Training Jobs.")
print(f"Output will land in: s3://{BUCKET}/hello-world/output")