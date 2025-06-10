"""
This module handles inference for the Airbnb occupancy model.
"""

import pandas as pd
import numpy as np
from joblib import load

# Load model and features
MODEL_PATH = "rf_model.joblib"
FEATURES_PATH = "rf_features.joblib"

model = load(MODEL_PATH)
feature_names = load(FEATURES_PATH)

def get_prediction(**kwargs) -> float:
    """
    Get occupancy rate prediction from input features.
    Expects inputs as keyword arguments.
    """
    input_df = pd.DataFrame([kwargs])[feature_names]
    prediction = model.predict(input_df)[0]
    return prediction * 100  # Convert to percentage

def get_top_features(top_n=15) -> pd.DataFrame:
    """
    Return the top N most important features.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1][:top_n]
        return pd.DataFrame({
            "Feature": [feature_names[i] for i in sorted_idx],
            "Importance": [importances[i] for i in sorted_idx]
        })
    return pd.DataFrame(columns=["Feature", "Importance"])
