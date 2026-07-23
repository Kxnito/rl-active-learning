"""
Hello-world SageMaker training script.

This is the script that actually runs INSIDE the SageMaker training container.
Its only job right now is to prove that:
  1. SageMaker can launch a job using our environment
  2. The job can read from S3 (via the input channel SageMaker mounts locally)
  3. The job can write results back out (SageMaker uploads /opt/ml/model and
     /opt/ml/output automatically to S3 when the job finishes)

Once this runs successfully for all 3 teammates, replace the body of main()
with real baseline / RL training code.
"""

import argparse
import os
import json
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    # SageMaker automatically sets these env vars / passes these as args
    # depending on how the job is launched. Defaults let this also run locally.
    parser.add_argument("--model-dir", type=str,
                         default=os.environ.get("SM_MODEL_DIR", "./output/model"))
    parser.add_argument("--output-data-dir", type=str,
                         default=os.environ.get("SM_OUTPUT_DATA_DIR", "./output/data"))
    parser.add_argument("--train-dir", type=str,
                         default=os.environ.get("SM_CHANNEL_TRAIN", "./data"))
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_data_dir, exist_ok=True)

    print("=== Hello World SageMaker Job ===")
    print(f"Timestamp: {datetime.utcnow().isoformat()}")
    print(f"Train data directory (mounted from S3): {args.train_dir}")

    # List whatever SageMaker mounted from S3 so we can confirm the data channel works
    if os.path.exists(args.train_dir):
        files = os.listdir(args.train_dir)
        print(f"Files found in train dir: {files}")
    else:
        print("No train dir found yet — that's fine for the very first test run.")

    # Write a small result file to prove we can save output back to S3
    result = {
        "status": "success",
        "message": "Hello from SageMaker! Environment and I/O are working.",
        "timestamp": datetime.utcnow().isoformat(),
    }
    result_path = os.path.join(args.output_data_dir, "hello_world_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote result to {result_path}")

    # Also "save a model" (just a placeholder file) to prove model-dir upload works
    model_path = os.path.join(args.model_dir, "hello_world_model.txt")
    with open(model_path, "w") as f:
        f.write("This is a placeholder model file.\n")
    print(f"Wrote placeholder model to {model_path}")

    print("=== Done ===")


if __name__ == "__main__":
    main()