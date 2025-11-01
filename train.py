import os
from joblib import load, dump
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_models(processed_dir="processed", models_dir="models"):
    # Load processed data
    print("Loading processed data...")
    X_train = load(os.path.join(processed_dir, "X_train.joblib"))
    X_test = load(os.path.join(processed_dir, "X_test.joblib"))
    y_train = load(os.path.join(processed_dir, "y_train.joblib"))
    y_test = load(os.path.join(processed_dir, "y_test.joblib"))

    os.makedirs(models_dir, exist_ok=True)

    print(" Training models...")

    # --- Ridge Regression ---
    print("\n Training Ridge model...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_preds = ridge.predict(X_test)
    ridge_mse = mean_squared_error(y_test, ridge_preds)
    print(f" Ridge MSE: {ridge_mse:.2f}")

    dump(ridge, os.path.join(models_dir, "ridge_model.joblib"))

    # --- Random Forest ---
    print("\n Training RandomForest model (faster mode)...")
    rf = RandomForestRegressor(
        n_estimators=100,  # Faster than 500
        max_depth=20,      # Reasonable limit
        n_jobs=-1,         # Use all cores
        random_state=42
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_mse = mean_squared_error(y_test, rf_preds)
    print(f" RandomForest MSE: {rf_mse:.2f}")

    dump(rf, os.path.join(models_dir, "random_forest_model.joblib"))

    print("\n Models saved in:", models_dir)
    print("\n Training complete!")


if __name__ == "__main__":
    train_models()
