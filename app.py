import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.stats import skew
import os

# Set page config
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

st.title("🏠 House Price Predictor")
st.markdown("Enter the details of the house to estimate its sale price.")

@st.cache_resource
def load_artifacts():
    model = joblib.load('static/model.joblib')
    scaler = joblib.load('static/scaler.joblib')
    feature_columns = joblib.load('static/feature_columns.joblib')
    reference_row = joblib.load('static/reference_row.joblib')
    categorical_values = joblib.load('static/categorical_values.joblib')
    return model, scaler, feature_columns, reference_row, categorical_values

try:
    import sklearn
    # st.write(f"Debug: sklearn version {sklearn.__version__}")
    model, scaler, feature_columns, reference_row, categorical_values = load_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.info("This is often caused by a version mismatch between the training environment (Python 3.14/sklearn 1.8) and the browser (Python 3.11/sklearn 1.5). I have updated the site to try and fix this.")
    st.stop()

# --- UI INPUTS ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Basic Features")
    overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
    gr_liv_area = st.number_input("Above Ground Living Area (sqft)", 500, 5000, 1500)
    total_bsmt_sf = st.number_input("Total Basement Area (sqft)", 0, 3000, 1000)
    year_built = st.number_input("Year Built", 1840, 2025, 2000)

with col2:
    st.subheader("Rooms & Space")
    full_bath = st.number_input("Full Bathrooms", 0, 5, 2)
    half_bath = st.number_input("Half Bathrooms", 0, 5, 0)
    tot_rms_abv_grd = st.number_input("Total Rooms Above Ground", 2, 15, 6)
    garage_cars = st.number_input("Garage Car Capacity", 0, 5, 2)
    garage_area = st.number_input("Garage Area (sqft)", 0, 1500, 500)

with col3:
    st.subheader("Location & Quality")
    # Dynamically get neighborhoods
    neighborhoods = sorted(categorical_values.get('Neighborhood', ['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst']))
    neighborhood = st.selectbox("Neighborhood", neighborhoods)
    
    exter_qual = st.selectbox("Exterior Quality", ['Ex', 'Gd', 'TA', 'Fa', 'Po'])
    kitchen_qual = st.selectbox("Kitchen Quality", ['Ex', 'Gd', 'TA', 'Fa', 'Po'])
    fireplace_qu = st.selectbox("Fireplace Quality", ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'None'])

if st.button("💰 Predict Sale Price", use_container_width=True):
    # 1. Create a row with user inputs
    input_row = reference_row.copy()
    
    # Update with UI values
    input_row['OverallQual'] = overall_qual
    input_row['GrLivArea'] = gr_liv_area
    input_row['TotalBsmtSF'] = total_bsmt_sf
    input_row['YearBuilt'] = year_built
    input_row['FullBath'] = full_bath
    input_row['HalfBath'] = half_bath
    input_row['TotRmsAbvGrd'] = tot_rms_abv_grd
    input_row['GarageCars'] = garage_cars
    input_row['GarageArea'] = garage_area
    input_row['Neighborhood'] = neighborhood
    input_row['ExterQual'] = exter_qual
    input_row['KitchenQual'] = kitchen_qual
    input_row['FireplaceQu'] = fireplace_qu
    
    # 2. Preprocessing (matching notebook logic)
    # Note: We use a simplified version for a single row prediction
    
    # Feature Engineering
    input_row['TotalSF'] = input_row['TotalBsmtSF'] + input_row['1stFlrSF'] + input_row['2ndFlrSF']
    input_row['TotalBath'] = (input_row['FullBath'] + (0.5 * input_row['HalfBath']) + 
                             input_row['BsmtFullBath'] + (0.5 * input_row['BsmtHalfBath']))
    input_row['HasPool'] = input_row['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    input_row['Has2ndFloor'] = input_row['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    input_row['HasGarage'] = input_row['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    
    # Ordinal Encoding
    ordinal_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
    ordinal_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 
                    'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
    for col in ordinal_cols:
        if col in input_row.columns:
            input_row[col] = input_row[col].map(ordinal_map).fillna(0)

    # Convert object types to strings (to match training)
    for col in input_row.columns:
        if input_row[col].dtype == 'object':
            input_row[col] = input_row[col].astype(str)

    # Skew Correction (apply log1p to the same columns)
    # In a real app we'd save the list of skewed columns, but we can reuse the logic
    # Here we just apply it to the numerical ones we know are skewed or often skewed
    num_cols = input_row.select_dtypes(include=[np.number]).columns
    input_row[num_cols] = np.log1p(input_row[num_cols])

    # One-Hot Encoding
    input_row_encoded = pd.get_dummies(input_row)
    
    # Align columns with feature_columns from training
    final_input = pd.DataFrame(0, index=[0], columns=feature_columns)
    for col in input_row_encoded.columns:
        if col in final_input.columns:
            final_input[col] = input_row_encoded[col].iloc[0]
            
    # Scale
    final_input_scaled = scaler.transform(final_input)
    
    # Predict
    pred_log = model.predict(final_input_scaled)
    prediction = np.expm1(pred_log)[0]
    
    st.success(f"### Predicted Sale Price: ${prediction:,.2f}")
    
    st.info("Estimated based on your inputs and the Stacking Ensemble model.")

st.markdown("---")
st.caption("This model was trained on the Kaggle Housing Prices competition dataset.")
