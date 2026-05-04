#!/usr/bin/env python3
"""Submit a new Kubeflow pipeline run with absolute paths."""
from kfp import Client

c = Client(host='http://127.0.0.1:8080')

# Submit with absolute paths for data files
run_result = c.create_run_from_pipeline_package(
    pipeline_file='outputs/pipeline.yaml',
    arguments={
        'transaction_path': '/workspace/data/train_transaction.csv',
        'identity_path': '/workspace/data/train_identity.csv',
    },
    namespace='fraud-detection'
)

print(f"NEW RUN SUBMITTED")
print(f"Run ID: {run_result.run_id}")
print(f"View at: http://127.0.0.1:8080/#/runs/details/{run_result.run_id}")
