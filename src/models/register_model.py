# mlflow.register_model(runs:/<run_id>/<model_path>,...) : The 'experiment_info.json' contains the 'run_id' and 'model_path' to register
#  the model. and Registration creates logged artifact a registered versioned model for model lifecycle management (versions, stages,
#  transitions, deployment etc.) in Mlflow Model Registry under the name "satya" that points to the run artifact that was stored into
#  the configured storage like s3 (it doesn’t copy another independent file by default — it references the artifact path).
# NOTE: experiment_info.json is required for code modularity.

# client.set_registered_model_alias(...) : Aliases are mutable pointers — you can point the alias at a new version when you promote a new
#  model. Use this when you want friendly names like "staging", "production", or "champion" to refer to specific versions. Aliases 
# (mutable named pointers): an alias name (e.g., "staging", "production_model", "champion") is a pointer to a single model version at a 
# time. When you call client.set_registered_model_alias(name, alias, version), you assign that alias to that version — that alias will 
# point to that single version. Reassigning the alias to a different version will make it point to the new version instead. So you cannot 
# have one alias name pointing to multiple versions simultaneously (an alias resolves to one version). staging as aliase means the model 
# from artifact as official model in the MLFLOW Model Registory.

# Example: you take that cake in the fridge and put it into the bakery to display case (Model Registry) so people can request it by 
# name/version/stage/alias.

import json, mlflow, logging, os
from dotenv import load_dotenv
load_dotenv()

# Set up MLflow tracking URI
mlflow.set_tracking_uri(os.getenv("satya_mlflow_ec2_uri"))


# logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('logs/model_registration_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry and assign version using run artifact URI that point to the AWS."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        
        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        
        # Transition the model to "Staging" stage from None
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(
            name=model_name,
            alias="staging",
            version=model_version.version
        )
        
        logger.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise

def main():
    try:
        model_info = load_model_info('logs/experiment_info.json')
        
        register_model('satya', model_info)
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()