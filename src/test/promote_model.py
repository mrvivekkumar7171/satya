# move the model from staging to production and archive the current production model.

import os, mlflow
from dotenv import load_dotenv
load_dotenv()

satya_mlflow_ec2_uri = os.getenv("satya_mlflow_ec2_uri")
staging_alias = "staging"
prod_alias = "production" # for initial nu
archived_alias = "archived"
model_name = os.getenv("FINAL_MODEL_NAME")

def promote_model():

    # Set up AWS MLflow tracking URI
    mlflow.set_tracking_uri(satya_mlflow_ec2_uri)
    client = mlflow.MlflowClient()

    # Get the latest version in staging
    latest_version_staging = client.get_model_version_by_alias(name=model_name, alias=staging_alias).version

    try:
        # Try to get current production model
        prod_versions = client.get_model_version_by_alias(name=model_name, alias=prod_alias).version
        # Archive it if exists
        client.set_registered_model_alias(name=model_name, version=prod_versions, alias=archived_alias)
        print(f"Archived previous Production model version {prod_versions}")
    except Exception as e:
        print(f"No production model found to Archive due to {e}")

    # Promote the new model to production
    client.set_registered_model_alias(name=model_name,version=latest_version_staging,alias=prod_alias)
    print(f"Model version {latest_version_staging} promoted to Production")

if __name__ == "__main__":
    promote_model()