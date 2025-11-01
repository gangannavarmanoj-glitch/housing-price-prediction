import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from joblib import dump

# Load data
data = pd.read_csv("data/train.csv")

# Select only the main useful features
features = ["LotArea", "OverallQual", "YearBuilt", "GrLivArea", "FullBath", "GarageCars"]
X = data[features]
y = data["SalePrice"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
dump(model, "models/simple_model.joblib")

print("Simple model trained and saved successfully in 'models/simple_model.joblib'.")
