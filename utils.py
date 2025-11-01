import os
import matplotlib.pyplot as plt
import joblib
import numpy as np

# ============================================================
# Utility Functions
# ============================================================

def save_object(obj, filename):
    """Save a Python object using joblib"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    joblib.dump(obj, filename)
    print(f"💾 Saved: {filename}")

def load_object(filename):
    """Load a Python object using joblib"""
    obj = joblib.load(filename)
    print(f"📂 Loaded: {filename}")
    return obj

def save_plot(y_true, y_pred, out_path, title="Prediction vs Actual"):
    """Save a scatter plot comparing actual vs predicted values"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.xlabel("Actual Sale Price")
    plt.ylabel("Predicted Sale Price")
    plt.title(title)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color="red")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"📊 Plot saved at: {out_path}")

def evaluate_regression(y_true, y_pred):
    """Return regression metrics (MAE, MSE, RMSE, R2)"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    metrics = {
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 3)
    }
    return metrics
