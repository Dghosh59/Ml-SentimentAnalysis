import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

from preproccessor import TextPreproccessor
from feature_extractor import CustomFeatureExtractor,VaderLexiconExtractor
from feature_extractor import PreprocessingTransformer


def build_pipeline():
    feature_union = FeatureUnion([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("custom", Pipeline([
            ("extract", CustomFeatureExtractor()),
            ("scale", StandardScaler())
        ])),
        ("lexicon", Pipeline([
            ("extract", VaderLexiconExtractor()),
            ("scale", StandardScaler())
        ]))
    ])

    pipeline = Pipeline([
        ("preprocess", PreprocessingTransformer()),  # your preprocessing
        ("features", feature_union),                 # feature extraction
        ("clf", LogisticRegression(max_iter=1000))   # classifier
    ])
    return pipeline


if __name__ == "__main__":
    df = pd.read_csv("new_data.csv")  
    
    pipeline = build_pipeline()
    pipeline.fit(df["Description"], df["Is_Response"])
    joblib.dump(pipeline, "sentiment_pipeline.pkl")
    print("✅ Model trained and saved as sentiment_pipeline.pkl")
