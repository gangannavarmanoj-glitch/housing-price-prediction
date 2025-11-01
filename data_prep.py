import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump
import os

def preprocess_data(input_csv, out_dir):
    # Load the dataset
    df = pd.read_csv(input_csv)
    print(f"✅ Data loaded successfully with shape: {df.shape}")

    # Drop rows with missing target
    df = df.dropna(subset=['SalePrice'])

    # Separate features and target
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']

    # Identify numeric and categorical columns
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    print(f"🔢 Numeric columns: {len(num_cols)}, 🏷️ Categorical columns: {len(cat_cols)}")

    # Convert all categorical values to strings
    for col in cat_cols:
        X[col] = X[col].astype(str)

    # Define transformers with Imputers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ]
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit preprocessor and transform
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    # Create output directory if it doesn’t exist
    os.makedirs(out_dir, exist_ok=True)

    # Save processed data and preprocessor
    dump(X_train_prep, os.path.join(out_dir, 'X_train.joblib'))
    dump(X_test_prep, os.path.join(out_dir, 'X_test.joblib'))
    dump(y_train, os.path.join(out_dir, 'y_train.joblib'))
    dump(y_test, os.path.join(out_dir, 'y_test.joblib'))
    dump(preprocessor, os.path.join(out_dir, 'preprocessor.joblib'))

    print(f"✅ Saved processed files in: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="data/train.csv", help="Path to input CSV file")
    parser.add_argument("--out_dir", default="processed", help="Output directory for processed files")
    args = parser.parse_args()

    preprocess_data(args.input_csv, args.out_dir)

