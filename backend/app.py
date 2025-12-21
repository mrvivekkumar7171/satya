# loading the model from model registry through mlflow and the vectorizer from local > user ask for prediction of comments > making prediction after applying preprocessing

# to run : python backend/app.py
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request, jsonify
from mlflow.tracking import MlflowClient
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import mlflow, joblib, re
from flask_cors import CORS
import pandas as pd
import pickle

satya_mlflow_ec2_uri = 'http://65.2.37.109:5000/'
FINAL_MODEL_NAME = 'satya'
MODEL_VERSION = '1'

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment

def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    """Load the saved TF-IDF vectorizer."""
    try:
        with open(vectorizer_path, 'rb') as file:
            vectorizer = joblib.load(file)
        return vectorizer
    except Exception as e:
        print(f'Error loading vectorizer from {vectorizer_path}: {e}')
        raise

# Load the model and vectorizer from the model registry and local storage
def load_model(model_name: str, model_version: str):
    # Set MLflow tracking URI to your server
    mlflow.set_tracking_uri(satya_mlflow_ec2_uri)  # Replace with your MLflow tracking URI
    client = MlflowClient()

    model_uri = f"models:/{model_name}/{model_version}"
    # f"models:/satya/staging"        or            f"runs:/{run_id}/{artifact_path}" 
    model = mlflow.pyfunc.load_model(model_uri) # to load the model from s3 using mlflow.
    return model

# load local model
def load_local_model(model_path: str):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        print(f'Error loading model from {model_path}: {e}')
        raise

# Initialize the model and vectorizer (model name version and vectorizer path can be configured)
# model = load_model(FINAL_MODEL_NAME, MODEL_VERSION)
model = load_local_model("./models/lgbm_model.pkl")  # Load the model from local not s3
vectorizer = load_vectorizer("./models/tfidf_vectorizer.pkl")  # Load the vectorizer from local not s3

@app.route('/health')
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')
    
    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        # Get feature names from the vectorizer
        feature_names = vectorizer.get_feature_names_out()

        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # Transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)
        
        # Convert sparse matrix to DataFrame
        transformed_comments = pd.DataFrame(transformed_comments.toarray(), columns=feature_names)

        # Make predictions
        predictions = model.predict(transformed_comments).tolist()  # Convert to list
        
        # Convert predictions to strings for consistency
        predictions = [str(pred) for pred in predictions]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    # Return the response with original comments and predicted sentiments
    response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) # for local testing