FROM python:3.11.13-slim

WORKDIR /app

# Install system dependencies (not python library) for parallel computing used by lightgbm
RUN apt-get update && apt-get install -y libgomp1

COPY backend/ /app/

# Create a models folder inside /app (empty if you don't copy anything now)
RUN mkdir -p /app/models

COPY models/tfidf_vectorizer.pkl /app/models/tfidf_vectorizer.pkl

RUN pip install -r requirements.txt

# downloading stopwords and wordnet from nltk
RUN python -m nltk.downloader stopwords wordnet

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]