pipeline {
    agent any

    // Environment Variables (For Docker Containers)
    environment {
        MLFLOW_TRACKING_URI    = 'http://mlflow_server:5000' // MLflow address within the Docker network
        MLFLOW_S3_ENDPOINT_URL = 'http://minio:9000'         // MinIO address within the Docker network
        AWS_ACCESS_KEY_ID      = 'minioadmin'
        AWS_SECRET_ACCESS_KEY  = 'strongpassword123'         // Your MinIO secret
        MINIO_DEPLOY_BUCKET    = 'mlops-test'
        MINIO_DEPLOY_PATH      = 'models'
        MLFLOW_EXPERIMENT_NAME = 'MobileNetV2-HapticMat-Quantized'
    }

    stages {
        stage('Setup') {
            steps {
                // 1. Create virtual environment
                sh 'python3 -m venv venv'

                // 2. Activate virtual environment and install packages
                sh 'venv/bin/pip install mlflow boto3 requests minio'
            }
        }

        stage('Get Latest Model from MLflow') {
            steps {
                script {
                    // 1) Find the latest FINISHED run_id using the Python API
                    def latestRunId = sh(
                            script: """
                            venv/bin/python - << 'EOF'
import os
from mlflow.tracking import MlflowClient

exp_name = os.environ.get("MLFLOW_EXPERIMENT_NAME")
if not exp_name:
    print("ERROR:NO_EXPERIMENT_NAME")
    raise SystemExit(0)

client = MlflowClient()

exp = client.get_experiment_by_name(exp_name)
if exp is None:
    print("ERROR:EXPERIMENT_NOT_FOUND")
    raise SystemExit(0)

runs = client.search_runs(
    [exp.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["attributes.start_time DESC"],
    max_results=1,
)

if not runs:
    print("ERROR:NO_FINISHED_RUN")
    raise SystemExit(0)

print(runs[0].info.run_id)
EOF
                        """,
                            returnStdout: true
                    ).trim()

                    // Catch error conditions
                    if (latestRunId.startsWith("ERROR:")) {
                        if (latestRunId == "ERROR:NO_EXPERIMENT_NAME") {
                            error("ERROR: The MLFLOW_EXPERIMENT_NAME environment variable is not defined.")
                        } else if (latestRunId == "ERROR:EXPERIMENT_NOT_FOUND") {
                            error("ERROR: MLflow experiment named '${env.MLFLOW_EXPERIMENT_NAME}' not found.")
                        } else if (latestRunId == "ERROR:NO_FINISHED_RUN") {
                            error("ERROR: No run with 'FINISHED' status found for this experiment.")
                        } else {
                            error("ERROR: An unexpected error occurred while searching for MLflow run: ${latestRunId}")
                        }
                    }

                    env.LATEST_RUN_ID = latestRunId
                    echo "Latest Successful Run ID: ${env.LATEST_RUN_ID}"

                    // 2) Download model and manifest files
                    def mlflow_cli = "venv/bin/mlflow"

                    sh """
                        ${mlflow_cli} artifacts download \
                            --run-id ${env.LATEST_RUN_ID} \
                            --artifact-path model_files/mv2_xnnpack.pte \
                            --dst-path .

                        ${mlflow_cli} artifacts download \
                            --run-id ${env.LATEST_RUN_ID} \
                            --artifact-path model_files/manifest.json \
                            --dst-path .
                    """
                }
            }
        }

        stage('Deploy to MinIO') {
            steps {
                script {
                    def modelFileName = 'mv2_xnnpack.pte'

                    // Do everything inside Python: read manifest.json, get SHA & version, upload to MinIO
                    sh """
venv/bin/python - << 'EOF'
import json
from datetime import datetime, timezone
from minio import Minio
from minio.error import S3Error

# 1) Read manifest.json
with open('manifest.json', 'r') as f:
    manifest = json.load(f)

new_sha256 = manifest.get('sha256')
version = manifest.get('version', 'unknown')

if not new_sha256:
    print("ERROR: 'sha256' field not found in manifest.json.")
    raise SystemExit(1)

# 2) MinIO Client
client = Minio(
    'minio:9000',  # Docker network address
    access_key='${env.AWS_ACCESS_KEY_ID}',
    secret_key='${env.AWS_SECRET_ACCESS_KEY}',
    secure=False
)

# 3) Copy model file
try:
    client.fput_object(
        '${env.MINIO_DEPLOY_BUCKET}',
        '${env.MINIO_DEPLOY_PATH}/' + '${modelFileName}',
        '${modelFileName}'
    )
    print("Model file successfully copied to MinIO.")
except S3Error as e:
    print(f"Model copy error: {e}")
    raise SystemExit(1)

# 4) Create and upload latest.json
latest_json_content = {
    'project': 'edgecomfort-ai',
    'version': version,
    'framework': 'executorch',
    'sha256': new_sha256,
    'objects': {
        'model': '${env.MINIO_DEPLOY_PATH}/' + '${modelFileName}'
    },
    'created_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S%z'),
    'notes': 'Deployed from Jenkins Pipeline Run ${env.LATEST_RUN_ID}'
}

with open('latest.json', 'w') as f:
    json.dump(latest_json_content, f, indent=4)

client.fput_object(
    '${env.MINIO_DEPLOY_BUCKET}',
    '${env.MINIO_DEPLOY_PATH}/latest.json',
    'latest.json'
)
print("latest.json file successfully uploaded to MinIO.")
EOF
                    """
                }
            }
        }
    }
}