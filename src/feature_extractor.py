import re
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER if not already present
#nltk.download('vader_lexicon')


# ----------- Custom Feature Extractor -----------
positive_words = {"good", "great", "awesome", "fantastic", "love", "excellent", "amazing"}
negative_words = {"bad", "terrible", "awful", "hate", "worst", "poor", "boring"}

class CustomFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        for text in X:
            tokens = text.lower().split()

            pos_count = sum(1 for w in tokens if w in positive_words)
            neg_count = sum(1 for w in tokens if w in negative_words)

            exclam_count = text.count("!")
            question_count = text.count("?")
            elongated_count = len(re.findall(r'(.)\1{2,}', text.lower()))
            word_count = len(tokens)
            char_count = len(text)
            uppercase_count = sum(1 for w in text.split() if w.isupper())

            features.append([
                pos_count,
                neg_count,
                exclam_count,
                question_count,
                elongated_count,
                word_count,
                char_count,
                uppercase_count
            ])
        return np.array(features)


# ----------- VADER Lexicon Extractor -----------
class VaderLexiconExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        for text in X:
            scores = self.vader.polarity_scores(text)
            features.append([
                scores['pos'], scores['neg'], scores['neu'], scores['compound']
            ])
        return np.array(features)

class PreprocessingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        from preproccessor import TextPreproccessor
        self.pre = TextPreproccessor()
    def fit(self, X, y=None): return self
    def transform(self, X):

        return [self.pre.preproccess(text) for text in X]
