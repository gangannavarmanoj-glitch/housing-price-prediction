import argparse
import os
from joblib import load
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json

def evaluate_model(model_path, X_test_path, y_test_path, out_dir):
    print("📊 Evaluating model...")

    # Load model and data
    model = load(model_path)
    X_test = load(X_test_path)
    y_test = load(y_test_path)

    # Predict
    preds = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    metrics = {
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 3)
    }

    print("✅ Evaluation complete!")
    print(json.dumps(metrics, indent=4))

    os.makedirs(out_dir, exist_ok=True)

    # Save metrics
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Plot: Actual vs Predicted
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds, alpha=0.5)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted House Prices")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "actual_vs_pred.png"))
    plt.close()

    # Plot: Residuals
    residuals = y_test - preds
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=30, edgecolor="black")
    plt.title("Residuals Distribution")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "residuals.png"))
    plt.close()

    print(f"📁 Results saved in: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model", required=True, help="Path to trained model file")
    parser.add_argument("--test_X", required=True, help="Path to processed X_test.joblib")
    parser.add_argument("--test_y", required=True, help="Path to processed y_test.joblib")
    parser.add_argument("--out_dir", default="reports", help="Output directory for reports")
    args = parser.parse_args()

    evaluate_model(args.model, args.test_X, args.test_y, args.out_dir)
