"""
This module automates model training for predicting Airbnb occupancy rate.
"""

import argparse
import pandas as pd
import numpy as np
import datetime
import logging
import json
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO)

def run(data_path, save_path):
    """
    Main script to perform model training.
        Parameters:
            data_path (str): Path to training dataset (CSV)
            save_path (str): Output folder path for saving model & metadata
    """
    logging.info("Loading data...")
    df = pd.read_csv(data_path)

    # Split features/label
    X = df.drop(columns=["occupancy_rate"])
    y = df["occupancy_rate"]

    feature_names = X.columns.tolist()

    # Train-Test split
    logging.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    logging.info("Training RandomForestRegressor...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
    rf.fit(X_train, y_train)

    # Evaluate
    logging.info("Evaluating model...")
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logging.info(f"Model performance - MSE: {mse:.4f}, R2: {r2:.4f}")

    # Save model and feature list
    model_filename = f"{save_path}/rf_model.joblib"
    features_filename = f"{save_path}/rf_features.joblib"
    metadata_filename = f"{save_path}/rf_model_metadata.json"

    dump(rf, model_filename)
    dump(feature_names, features_filename)

    metadata = {
        "name": "airbnb_occupancy_model",
        "metrics": f"mse:{mse:.4f}, r2:{r2:.4f}",
        "version": 1,
        "registration_date": str(datetime.datetime.now()),
        "model": model_filename,
        "features": features_filename
    }

    with open(metadata_filename, "w") as f:
        json.dump(metadata, f, indent=4)

    logging.info(f"Model and metadata saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to df_pred.csv")
    parser.add_argument("--save_path", type=str, default=".", help="Directory to save model files")
    args = parser.parse_args()

    run(args.data_path, args.save_path)