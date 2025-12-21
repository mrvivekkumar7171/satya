# to test MLflow tracking server connection and logging functionality before running or making changes in the model_evaluation.py

import mlflow, random, os
satya_mlflow_ec2_uri = 'http://65.2.37.109:5000/'

# Set the MLflow tracking URI
mlflow.set_tracking_uri(satya_mlflow_ec2_uri)

# Start an MLflow run
with mlflow.start_run():
    # Log some random parameters
    mlflow.log_param("param1", random.randint(1, 100))
    mlflow.log_param("param2", random.random())

    # Log some random metrics
    mlflow.log_metric("metric1", random.random())
    mlflow.log_metric("metric2", random.uniform(0.5, 1.5))

    print("Logged random parameters and metrics.")