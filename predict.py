import argparse
import pandas as pd
from joblib import load
import os

def predict_prices(model_path, preprocessor_path, input_csv, out_dir):
    print("📈 Loading model and preprocessor...")

    # Load model and preprocessor
    model = load(model_path)
    preprocessor = load(preprocessor_path)

    print("📄 Loading input data...")
    new_data = pd.read_csv(input_csv)
    print(f"✅ Loaded new data with shape: {new_data.shape}")

    # Convert categorical columns to string to avoid mixed types
    cat_cols = new_data.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        new_data[col] = new_data[col].astype(str)

    # Apply preprocessing
    X_new = preprocessor.transform(new_data)

    # Predict
    predictions = model.predict(X_new)

    # Save results
    os.makedirs(out_dir, exist_ok=True)
    results = new_data.copy()
    results["Predicted_SalePrice"] = predictions

    output_file = os.path.join(out_dir, "predictions.csv")
    results.to_csv(output_file, index=False)

    print(f"✅ Predictions saved to: {output_file}")
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict house prices on new data")
    parser.add_argument("--model", required=True, help="Path to trained model file")
    parser.add_argument("--preprocessor", required=True, help="Path to saved preprocessor file")
    parser.add_argument("--input_csv", required=True, help="Path to new data CSV file")
    parser.add_argument("--out_dir", default="predictions", help="Directory to save predictions")
    args = parser.parse_args()

    predict_prices(args.model, args.preprocessor, args.input_csv, args.out_dir)
