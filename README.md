# price-prediction-for-airbnb-listings
This project was completed as part of a technical interview assignment for Spark Tech Agency.
This project builds a machine learning model to predict Airbnb listing prices using structured tabular data. The workflow includes data cleaning, exploratory analysis, model evaluation, hyperparameter tuning, and feature engineering.
## Project Overview
- Dataset: Airbnb listings with features like price, rating, check-in/out policies, country, etc.
- Goal: Predict listing price based on available features.
## Workflow
1. **Exploratory Data Analysis (EDA)**
   - Analyzed distributions, correlations, and outliers.
   - Noted skewness in price and other numeric features.
2.  **Data Cleaning**  
    - Handled missing values and outliers using IQR.
    - One-hot encoded categorical columns (`country`).
3. **Modeling Approach**
   - Baseline: Feedforward Neural Network (MAE ~4738)
   - Improved: Random Forest and XGBoost (default)
   - Tuned: XGBoost with `RandomizedSearchCV` (MAE ~4006)

4. **Feature Engineering**
   - Added `price_per_rating`, `checkin_flex`, and `checkout_flex`
   - Final MAE (5-fold CV): **~208.72**
   - Test Set MAE: **192.70**
## Technologies Used
- Python, NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn, XGBoost, TensorFlow (Keras)
- Jupyter Notebook / Python Script
## Future Work
- Deploy the Model as an API or Web App
- Monitor Model Drift Over Time
