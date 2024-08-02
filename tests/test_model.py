import unittest
import mlflow
import os
import pandas as pd
from mlflow.pyfunc import PyFuncModel

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up DagsHub credentials for MLflow tracking
        dagshub_token = os.getenv("DAGSHUB_PAT")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "campusx-official"
        repo_name = "mlops-project-2"

        # Set up MLflow tracking URI
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        # Load the model from MLflow model registry
        cls.model_name = "my_model"
        cls.model_version = cls.get_latest_model_version(cls.model_name)
        cls.model_uri = f'models:/{cls.model_name}/{cls.model_version}'
        cls.model = mlflow.pyfunc.load_model(cls.model_uri)

    @staticmethod
    def get_latest_model_version(model_name):
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=["Staging"])
        return latest_version[0].version if latest_version else None

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.model)

    def test_model_signature(self):
        # Assuming the input data has the shape (number of samples, number of features)
        # Create a dummy input for the model based on expected input shape
        input_data = pd.DataFrame({
            "feature1": [0],
            "feature2": [0],
            "feature3": [0],
            # Add all required features here
        })

        # Predict using the model to verify the input and output shapes
        prediction = self.model.predict(input_data)

        # Verify the input shape
        self.assertEqual(input_data.shape[1], len(self.model.metadata.get_input_schema().columns))
        
        # Verify the output shape (assuming binary classification with a single output)
        self.assertEqual(len(prediction), len(input_data))
        self.assertEqual(prediction.shape[1], 1)  # Assuming a single output column

if __name__ == "__main__":
    unittest.main()
