from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import sys

# make sure feature_extractors.py is importable
sys.path.append(os.path.dirname(__file__))

from feature_extractor import PreprocessingTransformer, CustomFeatureExtractor, VaderLexiconExtractor

# Load trained pipeline
pipeline = joblib.load(os.path.join(os.path.dirname(__file__), "sentiment_pipeline.pkl"))

# FastAPI app
app = FastAPI(title="Sentiment Analysis API", version="1.0")

# Request body schema
class SentimentRequest(BaseModel):
    text: str

# Root endpoint
@app.get("/")
def root():
    return {"message": "Sentiment Analysis API is running 🚀"}

# Prediction endpoint
@app.post("/predict")
def predict_sentiment(request: SentimentRequest):
    text = request.text
    prediction = pipeline.predict([text])[0]
    proba = pipeline.predict_proba([text])[0].tolist()

    return {
        "text": text,
        "predicted_label": int(prediction),
        "probabilities": proba
    }

