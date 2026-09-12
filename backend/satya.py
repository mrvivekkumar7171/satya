import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

from sklearn.feature_extraction.text import TfidfVectorizer
import mlflow, joblib, io, re, os, requests
from flask import app, jsonify, send_file
from mlflow.tracking import MlflowClient
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import pickle

load_dotenv()
MODEL_VERSION = os.getenv('MODEL_VERSION')
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
FINAL_MODEL_NAME = os.getenv('FINAL_MODEL_NAME')
slice_limit = int(os.getenv('slice_limit'))
max_comments = int(os.getenv('max_comments'))
vectorizer_path = os.getenv('vectorizer_path')
local_model_path = os.getenv('local_model_path')
satya_mlflow_ec2_uri = os.getenv('satya_mlflow_ec2_uri')

def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    """Load the saved TF-IDF vectorizer."""
    try:
        with open(vectorizer_path, 'rb') as file:
            vectorizer = joblib.load(file)
        return vectorizer
    except Exception as e:
        print(f'Error loading vectorizer from {vectorizer_path}: {e}')
        raise

def load_model(model_name: str, model_version: str):
    """ Load the model and vectorizer from the model registry and local storage"""
    # Set MLflow tracking URI to your server
    mlflow.set_tracking_uri(satya_mlflow_ec2_uri)  # Replace with your MLflow tracking URI
    client = MlflowClient()

    model_uri = f"models:/{model_name}/{model_version}"
    # f"models:/satya/staging"        or            f"runs:/{run_id}/{artifact_path}" 
    model = mlflow.pyfunc.load_model(model_uri) # to load the model from s3 using mlflow.
    return model

def load_local_model(model_path: str):
    """load local model"""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        print(f'Error loading model from {model_path}: {e}')
        raise

# model = load_model(FINAL_MODEL_NAME, MODEL_VERSION)
model = load_local_model(local_model_path)
vectorizer = load_vectorizer(vectorizer_path)

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

def analyze_youtube_video(video_id):
    """
    Fetches comments for a video_id, predicts sentiment, and returns formatted data.
    """
    comments_url = "https://www.googleapis.com/youtube/v3/commentThreads"
    comments_data = []
    page_token = ""
    

    # 1. Fetch Comments from YouTube (Max 500)
    try:
        while len(comments_data) < max_comments:
            params = {
                'part': 'snippet',
                'videoId': video_id,
                'maxResults': slice_limit, # YouTube API max results per request
                'key': YOUTUBE_API_KEY,
                'pageToken': page_token,
                'textFormat': 'plainText'
            }
            response = requests.get(comments_url, params=params)
            data = response.json()
            
            # Check for API Errors (Like Comments Disabled)
            if 'error' in data:
                errors = data['error'].get('errors', [])
                if errors:
                    reason = errors[0].get('reason')
                    if reason == 'commentsDisabled':
                        return {"error": "Comments are disabled for this video."}
                    if reason == 'videoNotFound':
                        return {"error": "Video not found."}
                    if reason == 'quotaExceeded':
                        return {"error": "API Quota exceeded. Please try again later."}

            for item in data['items']:
                comment_snippet = item['snippet']['topLevelComment']['snippet']
                comments_data.append({
                    'text': comment_snippet['textOriginal'],
                    'timestamp': comment_snippet['publishedAt'],
                    'authorId': comment_snippet.get('authorChannelId', {}).get('value', 'Unknown')
                })
            
            page_token = data.get('nextPageToken')
            if not page_token:
                break
                
    except Exception as e:
        print(f"Error fetching YouTube comments: {e}")
        return {"error": str(e)}

    if not comments_data:
        return {"error": "No comments found for the provided video ID."}


    # 2. ML Prediction Logic
    try:
        # Preprocess
        preprocessed_comments = [preprocess_comment(comment['text']) for comment in comments_data]

        # Vectorize
        feature_names = vectorizer.get_feature_names_out()
        transformed_comments = vectorizer.transform(preprocessed_comments)
        transformed_comments = pd.DataFrame(transformed_comments.toarray(), columns=feature_names)

        # Predict Class
        predictions = model.predict(transformed_comments)
        
        # Predict Confidence (Probability) Note: If your model supports predict_proba, use it. Otherwise default to 1.0
        try:
            probs = model.predict_proba(transformed_comments) # why not .tolist() here
            confidence_scores = np.max(probs, axis=1).tolist()
        except AttributeError:
            confidence_scores = [1.0] * len(predictions)


        # 3. Format Response
        formatted_response = []
        for i, (comment_obj, pred, conf) in enumerate(zip(comments_data, predictions, confidence_scores)):
            formatted_response.append({
                "Original_Comment": comment_obj['text'], # sting
                "Processed_Comment": preprocessed_comments[i], # sting
                "confidence": round(float(conf), 2),
                "sentiment": int(pred),
                "timestamp": comment_obj['timestamp'], # sting
                "AuthorID": comment_obj['authorId'] # sting
            })
            
        return formatted_response

    except Exception as e:
        print(f"Error in analysis: {e}")
        return {"error": f"Prediction failed: {str(e)}"}

def predict_sentiment(comments):
    # Preprocess each comment before vectorizing
    preprocessed_comments = [preprocess_comment(comment) for comment in comments]

    # Transform comments using the vectorizer
    transformed_comments = vectorizer.transform(preprocessed_comments)

    # Get feature names from the vectorizer
    feature_names = vectorizer.get_feature_names_out()
        
    # Convert sparse matrix to DataFrame
    transformed_comments = pd.DataFrame(transformed_comments.toarray(), columns=feature_names)

    # Make predictions
    predictions = model.predict(transformed_comments).tolist()  # Convert to list
        
    # Convert predictions to strings for consistency
    predictions = [str(pred) for pred in predictions]
    return predictions