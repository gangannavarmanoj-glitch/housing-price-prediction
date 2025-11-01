import argparse
from joblib import load, dump
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import os

def train_models(X_train_path, y_train_path, out_dir):
    print("🏋️ Training models...")

    # Load data
    X_train = load(X_train_path)
    y_train = load(y_train_path)

    # Define candidate models
    models = {
        "Ridge": Ridge(),
        "RandomForest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42, objective='reg:squarederror')
    }

    # Define hyperparameter grids
    params = {
        "Ridge": {"alpha": [0.1, 1.0, 10.0]},
        "RandomForest": {"n_estimators": [100, 200], "max_depth": [10, 20]},
        "XGBoost": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]}
    }

    os.makedirs(out_dir, exist_ok=True)
    best_model = None
    best_score = float("inf")
    best_name = ""

    # ✅ Train and evaluate each model (properly indented)
    for name, model in models.items():
        print(f"Training {name} model...")

        grid = GridSearchCV(model, params[name], cv=5, scoring="neg_mean_squared_error")
        grid.fit(X_train, y_train)

        best_model_for_this = grid.best_estimator_
        best_score_for_this = -grid.best_score_
        print(f"{name} best score (MSE): {best_score_for_this:.4f}")

        # Save model
        dump(best_model_for_this, os.path.join(out_dir, f"{name}_model.joblib"))

        # Track best model overall
        if best_score_for_this < best_score:
            best_model = best_model_for_this
            best_score = best_score_for_this
            best_name = name

    print(f"\nBest model: {best_name} with MSE {best_score:.4f}")
    dump(best_model, os.path.join(out_dir, "best_model.joblib"))

# ✅ Add main block so you can run it directly
if __name__ == "__main__":
    train_models("processed/X_train.joblib", "processed/y_train.joblib", "models/")
