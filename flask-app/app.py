from flask import Flask, render_template,request
import mlflow
from preprocessing_utility import normalize_text
import dagshub
import pickle

mlflow.set_tracking_uri('https://dagshub.com/campusx-official/mlops-project-2.mlflow')
dagshub.init(repo_owner='campusx-official', repo_name='mlops-project-2', mlflow=True)

app = Flask(__name__)

# load model from model registry
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

vectorizer = pickle.load(open('models/vectorizer.pkl','rb'))



@app.route('/')
def home():
    return render_template('index.html',result=None)

@app.route('/predict', methods=['POST'])
def predict():

    text = request.form['text']

    # clean
    text = normalize_text(text)

    # bow
    features = vectorizer.transform([text])

    # prediction
    result = model.predict(features)

    # show
    return render_template('index.html', result= result[0])

app.run(debug=True)