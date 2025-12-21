# test if model is loading correctly or not from the s3 through MLflow model registry before deployment
from mlflow.tracking import MlflowClient
import mlflow.pyfunc
import pytest, os

satya_mlflow_ec2_uri = 'http://65.2.37.109:5000/'
FINAL_MODEL_NAME = 'satya'

# Set your remote tracking URI
mlflow.set_tracking_uri(satya_mlflow_ec2_uri)

@pytest.mark.parametrize("model_name, alias", [
    (FINAL_MODEL_NAME, "staging"),])
def test_load_latest_staging_model(model_name, alias):
    client = MlflowClient()
    
    # Get the latest version in the specified stage
    latest_version_info = client.get_model_version_by_alias(name=model_name, alias=alias)
    latest_version = latest_version_info.version if latest_version_info else None
    
    assert latest_version is not None, f"No model found in the '{alias}' stage for '{model_name}'"

    try:
        # Load the latest version of the model
        model_uri = f"models:/{model_name}/{latest_version}"
        model = mlflow.pyfunc.load_model(model_uri)

        # Ensure the model loads successfully
        assert model is not None, "Model failed to load"
        print(f"Model '{model_name}' version {latest_version} loaded successfully from '{alias}' stage.")

    except Exception as e:
        pytest.fail(f"Model loading failed with error: {e}")