🏡 House Price Prediction using Machine Learning
Predicting house prices using data from the Kaggle "House Prices - Advanced Regression Techniques" dataset.

Project Overview

This project aims to predict the sale prices of houses using advanced regression techniques.
It involves data preprocessing, feature engineering, model training, evaluation, and prediction.
The pipeline is fully automated — from raw data to Kaggle submission file.

Project Structure

housing-price-prediction/
├── data/                 # Contains training and test CSV files  
├── models/               # Trained models (.joblib)  
├── processed/            # Processed data and preprocessor file  
├── reports/              # Generated predictions and submission files  
├── venv/                 # Virtual environment (excluded from GitHub)
│
├── data_prep.py          # Script for data cleaning and feature processing  
├── train.py              # Script to train the ML model  
├── evaluate.py           # Evaluate model performance  
├── predict.py            # Predicts prices for test data  
├── make_submission.py    # Generates Kaggle submission file  
├── utils.py              # Helper functions  
├── requirements.txt      # Python dependencies  
└── README.md             # Project documentation

How to Run the Project

Step 1: Data Preparation
python data_prep.py

Step 2: Model Training
python train.py

Step 3: Evaluate the Model
python evaluate.py

Step 4: Make Predictions
python predict.py --model models/best_model.joblib --preprocessor processed/preprocessor.joblib --input_csv data/test.csv --out_dir reports

Step 5: Generate Kaggle Submission File
python make_submission.py

Model Details

Algorithm Used: Random Forest Regressor (can be swapped with XGBoost, etc.)
Evaluation Metric: Root Mean Squared Error (RMSE)
Preprocessing: Handling missing values, feature scaling, encoding categorical features

Outputs

reports/predictions.csv → model predictions
reports/submission.csv → ready for Kaggle submission

Dependencies

Main libraries used:
pandas
numpy
scikit-learn
joblib
matplotlib (optional for visualization)

Author

Manoj Gangannavar
Machine Learning Enthusiast | Data Science Learner
GitHub: @gangannavarmanoj-glitch


Future Improvements

Try XGBoost or LightGBM for better accuracy
Add feature selection and hyperparameter tuning
Build a Streamlit web app for interactive predictions
