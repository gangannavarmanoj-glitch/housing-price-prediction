import pandas as pd
from joblib import load

def main():
    print("House Price Prediction")
    print("-----------------------------------")

    # User inputs
    try:
        lot_area = float(input("Enter LotArea (e.g., 8500): "))
        overall_qual = int(input("Enter Overall Quality (1-10): "))
        year_built = int(input("Enter Year Built (e.g., 2005): "))
        gr_liv_area = float(input("Enter Above Ground Living Area (sq ft): "))
        full_bath = int(input("Enter Number of Full Bathrooms: "))
        garage_cars = int(input("Enter Garage Capacity (cars): "))
    except ValueError:
        print("Invalid input! Please enter valid numeric values.")
        return

    # Prepare input data
    new_data = pd.DataFrame([{
        "LotArea": lot_area,
        "OverallQual": overall_qual,
        "YearBuilt": year_built,
        "GrLivArea": gr_liv_area,
        "FullBath": full_bath,
        "GarageCars": garage_cars
    }])

    print("\nLoading model...")
    model = load("models/simple_model.joblib")

    # Predict
    prediction = model.predict(new_data)[0]
    print(f"\nPredicted House Price: ${prediction:,.2f}")

if __name__ == "__main__":
    main()

