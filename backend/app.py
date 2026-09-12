# python backend/app.py
from flask import Flask, render_template, request, jsonify
import satya
import os

app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/')
def SentimentAnalyzer():
    """Render the main sentiment analyzer page."""
    return render_template('index.html', title="Sentiment Analyzer")

@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    """Analyze the sentiment of comments for a given YouTube video."""
    video_id = request.get_json().get('videoId')
    results = satya.analyze_youtube_video(video_id)
    
    if isinstance(results, dict) and "error" in results:
        return jsonify(results), 500
        
    return jsonify(results)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = request.json.get('comments')
    
    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        predictions = satya.predict_sentiment(comments)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    # Return the response with original comments and predicted sentiments
    response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]
    return jsonify(response)

if __name__=="__main__":
    app.run(host='0.0.0.0', port=8000, debug=True) # debug=True is used to load pages on changes and only for development environment