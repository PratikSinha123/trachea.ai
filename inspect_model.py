import joblib
import os
import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class TextCombiner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return X

class FraudKeywordFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return X

class AdvancedLinguisticFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return X

class DeepHeuristicFlags(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return X

# Inject into __main__
import __main__
__main__.TextCombiner = TextCombiner
__main__.FraudKeywordFeatures = FraudKeywordFeatures
__main__.AdvancedLinguisticFeatures = AdvancedLinguisticFeatures
__main__.DeepHeuristicFlags = DeepHeuristicFlags

model_path = "/Users/pratiksinha/Spot-The-Scam-AI/models/model.pkl"
if os.path.exists(model_path):
    print(f"Inspecting {model_path}...")
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully!")
        print("Pipeline steps:", model.steps)
        if hasattr(model, 'named_steps') and 'features' in model.named_steps:
             print("Feature transformer:", model.named_steps['features'])
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print("Model not found.")
