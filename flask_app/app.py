from flask import Flask, render_template, request
import mlflow
from flask_app.preprocessing_utility import normalize_text
import dagshub
import pickle
import os
import pandas as pd

mlflow.set_tracking_uri('https://dagshub.com/campusx-official/mlops-project-2.mlflow')
dagshub.init(repo_owner='campusx-official', repo_name='mlops-project-2', mlflow=True)

app = Flask(__name__)

# Load model from model registry
def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["Production"])
    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages=["None"])
    return latest_version[0].version if latest_version else None

model_name = "my_model"
model_version = get_latest_model_version(model_name)

model_uri = f'models:/{model_name}/{model_version}'
model = mlflow.pyfunc.load_model(model_uri)

# Get the model's input schema
input_schema = model.metadata.get_input_schema()

vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']

    # Clean
    text = normalize_text(text)

    # Bag of Words
    features = vectorizer.transform([text])

    # Convert sparse matrix to DataFrame with correct column names
    features_df = pd.DataFrame.sparse.from_spmatrix(features, columns=input_schema.input_names())

    # Prediction
    result = model.predict(features_df)

    # Show
    return render_template('index.html', result=result[0])

if __name__ == '__main__':
    app.run(debug=True)
