import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.stats import skew
import plotly.express as px
import os

# Set page config
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load('static/model.joblib')
    scaler = joblib.load('static/scaler.joblib')
    feature_columns = joblib.load('static/feature_columns.joblib')
    reference_row = joblib.load('static/reference_row.joblib')
    categorical_values = joblib.load('static/categorical_values.joblib')
    return model, scaler, feature_columns, reference_row, categorical_values

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('./home-data-for-ml-course/train.csv')
        return df
    except:
        return None

# --- SIDEBAR ---
st.sidebar.header("🏆 Model Performance")
col_perf1, col_perf2 = st.sidebar.columns(2)
col_perf1.metric("RMSLE", "0.115")
st.sidebar.info("🎯 This score ranks in the **Top 2% to 5%** of the Kaggle leaderboard.")
st.sidebar.divider()

# --- PROJECT OVERVIEW ---
st.sidebar.header("📋 Project Overview")
st.sidebar.markdown(
    """
    This predictor is built using the **Ames Housing Dataset** from the [Kaggle House Prices Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).
    
    ### 🧬 The Model: Stacking Ensemble
    Instead of a single algorithm, we use a **Stacking Regressor** that aggregates four diverse models:
    1.  **XGBoost**
    2.  **Gradient Boosting**
    3.  **Ridge Regression**
    4.  **Lasso Regression**
    
    ### 🛠️ Preprocessing Pipeline
    - **Outlier Removal**
    - **Log Transformation**
    - **Feature Engineering**
    - **Ordinal Encoding**
    - **Robust Scaling**
    """
)

st.sidebar.divider()
st.sidebar.subheader("💻 Technical Stack")
st.sidebar.markdown(
    """
    - **ML:** Scikit-Learn, XGBoost
    - **Data:** Pandas, NumPy, Plotly
    - **Runtime:** Stlite (Wasm)
    """
)

try:
    model, scaler, feature_columns, reference_row, categorical_values = load_artifacts()
    raw_data = load_data()
except Exception as e:
    st.error(f"### Error loading model artifacts")
    st.code(str(e))
    st.stop()

# --- MAIN APP TABS ---
tab1, tab2, tab3 = st.tabs(["💰 Price Predictor", "📈 Data Insights (EDA)", "🧠 Model Insights"])

with tab1:
    st.title("🏠 House Price Predictor")
    st.markdown("Enter the details of the house to estimate its sale price.")
    st.markdown("## 📊 Kaggle Dataset Features")
    st.info("These interactive fields are mapped directly to the original [Kaggle Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) dataset.")

    st.markdown("### 🎛️ Configure Property Attributes")
    
    # Input zones
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🏗️ Structure & Quality")
        overall_qual = st.slider("Overall Quality", 1, 10, 6, help="1=Poor, 10=Excellent")
        gr_liv_area = st.number_input("Living Area (sqft)", 300, 6000, 1500, step=50)
        total_bsmt_sf = st.number_input("Basement Area (sqft)", 0, 4000, 1000, step=50)
        year_built = st.number_input("Year Built", 1870, 2025, 2000, step=1)

    with col2:
        st.subheader("🛌 Rooms & Space")
        full_bath = st.number_input("Full Bathrooms", 0, 5, 2, step=1)
        half_bath = st.number_input("Half Bathrooms", 0, 3, 0, step=1)
        tot_rms_abv_grd = st.number_input("Total Rooms", 2, 15, 6, step=1)
        garage_cars = st.number_input("Garage Capacity (Cars)", 0, 5, 2, step=1)
        garage_area = st.number_input("Garage Area (sqft)", 0, 1500, 500, step=50)

    with col3:
        st.subheader("📍 Location & Finish")
        neighborhoods = sorted(categorical_values.get('Neighborhood', []))
        # Ensure 'NAmes' is a valid option before setting it as default
        default_neighborhood_index = neighborhoods.index('NAmes') if 'NAmes' in neighborhoods else 0
        neighborhood = st.selectbox("Neighborhood", neighborhoods, index=default_neighborhood_index)
        exter_qual = st.selectbox("Exterior Quality", ['Ex', 'Gd', 'TA', 'Fa', 'Po'], index=2)
        kitchen_qual = st.selectbox("Kitchen Quality", ['Ex', 'Gd', 'TA', 'Fa', 'Po'], index=2)
        fireplace_qu = st.selectbox("Fireplace Quality", ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'None'], index=5)

    if st.button("💰 Predict Sale Price", use_container_width=True):
        # Nonsensical data validation
        if gr_liv_area <= 0 or total_bsmt_sf < 0:
            st.error("Invalid dimensions: Living Area and Basement Area must be positive.")
        else:
            try:
                # Create a row with user inputs
                input_row = reference_row.copy()
                
                # Map values
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

                for col in input_row.columns:
                    if input_row[col].dtype == 'object':
                        input_row[col] = input_row[col].astype(str)

                num_cols = input_row.select_dtypes(include=[np.number]).columns
                input_row[num_cols] = np.log1p(input_row[num_cols])

                input_row_encoded = pd.get_dummies(input_row)
                final_input = pd.DataFrame(0, index=[0], columns=feature_columns)
                for col in input_row_encoded.columns:
                    if col in final_input.columns:
                        final_input[col] = input_row_encoded[col].iloc[0]
                        
                final_input_scaled = scaler.transform(final_input)
                pred_log = model.predict(final_input_scaled)
                prediction = np.expm1(pred_log)[0]
                
                st.success(f"### Predicted Sale Price: ${prediction:,.2f}")
                st.balloons()
                
            except Exception as e:
                st.error(f"Prediction Error: {e}")

with tab2:
    st.header("📈 Exploratory Data Analysis")
    if raw_data is not None:
        col_eda1, col_eda2 = st.columns(2)
        
        with col_eda1:
            st.subheader("Price Distribution")
            fig_dist = px.histogram(raw_data, x="SalePrice", nbins=50, 
                                   title="Distribution of House Prices",
                                   color_discrete_sequence=['#3d85c6'],
                                   labels={'SalePrice': 'Sale Price ($)'})
            st.plotly_chart(fig_dist, use_container_width=True)
            
        with col_eda2:
            st.subheader("Quality vs Price")
            fig_scatter = px.box(raw_data, x="OverallQual", y="SalePrice",
                                title="Overall Quality Impact on Price",
                                color="OverallQual",
                                labels={'OverallQual': 'Overall Quality (1-10)', 'SalePrice': 'Sale Price ($)'})
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        st.subheader("Feature Correlation")
        numeric_df = raw_data.select_dtypes(include=[np.number])
        corr = numeric_df.corr()['SalePrice'].sort_values(ascending=False).head(12)
        fig_corr = px.bar(x=corr.index, y=corr.values, 
                         title="Top 12 Features Correlated with Sale Price",
                         labels={'x': 'Feature', 'y': 'Correlation Coefficient'},
                         color=corr.values, color_continuous_scale='Blues')
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("Training data not available for visualization. Check if 'train.csv' is correctly linked.")

with tab3:
    st.header("🧠 Model Explainability")
    st.markdown("""
    This section provides transparency into how the **Stacking Ensemble** arrives at a price estimation. 
    The model combines insights from linear and non-linear algorithms.
    """)
    
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        st.subheader("Expert Influence")
        try:
            meta_model = model.final_estimator_
            base_model_names = [name.upper() for name, _ in model.estimators]
            
            if hasattr(meta_model, 'coef_'):
                coefs = np.abs(meta_model.coef_)
                # Normalize to percentages
                influence = (coefs / np.sum(coefs)) * 100
                
                importance_df = pd.DataFrame({
                    'Expert Model': base_model_names,
                    'Influence (%)': influence
                })
                
                fig_importance = px.pie(importance_df, values='Influence (%)', names='Expert Model',
                                       title="Weightage of Base Algorithms",
                                       color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_importance, use_container_width=True)
            else:
                st.info("The meta-model does not use linear coefficients for weighting.")
        except Exception:
            st.info("Individual model weights are combined dynamically.")

    with col_exp2:
        st.subheader("Top Predictors")
        st.info("""
        **What drives the price?**
        - **GrLivArea**: The total living area above ground is the #1 physical driver.
        - **OverallQual**: High-end finishes and quality construction add significant premium.
        - **TotalSF**: The total footprint (including basement) captures the asset's scale.
        - **Neighborhood**: Location remains a primary factor in real estate valuation.
        """)
        
        st.markdown("### Model Complexity")
        st.write("""
        By stacking **XGBoost** (gradient boosting) with **Lasso/Ridge** (regularized linear models), 
        the ensemble captures both subtle non-linear interactions and broad linear trends simultaneously.
        """)

st.markdown("---")
st.caption("Data: Kaggle Ames Housing Dataset")
