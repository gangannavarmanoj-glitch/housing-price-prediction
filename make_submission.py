import pandas as pd

# Load the predictions file
df = pd.read_csv("reports/predictions.csv")

# If predictions were saved separately (e.g., in a model output array)
# check if the last column has predicted prices
if "Predicted_SalePrice" not in df.columns:
    print("⚠️ 'Predicted_SalePrice' column not found. Make sure your predict.py script saves predictions.")
else:
    # Prepare Kaggle submission format
    submission = df[["Id", "Predicted_SalePrice"]].rename(columns={"Predicted_SalePrice": "SalePrice"})
    
    # Save the submission file
    submission.to_csv("reports/submission.csv", index=False)
    print("✅ Kaggle submission file saved: reports/submission.csv")
