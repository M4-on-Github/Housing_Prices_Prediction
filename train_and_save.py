import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV, LassoCV
from xgboost import XGBRegressor
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

def preprocess_data(train_features, test_features):
    # Combine for uniform processing
    all_data = pd.concat([train_features, test_features]).reset_index(drop=True)

    # 4. CLEANING & FEATURE ENGINEERING
    # A. Impute LotFrontage by Neighborhood
    all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median()))

    # B. Fix specific integer columns that are actually categorical
    all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
    all_data['GarageYrBlt'] = all_data['GarageYrBlt'].fillna(0)

    # C. Fill all other missing values
    objects = []
    numerics = []
    for c in all_data.columns:
        if all_data[c].dtype == 'object':
            objects.append(c)
        else:
            numerics.append(c)

    all_data[objects] = all_data[objects].fillna("None")
    all_data[numerics] = all_data[numerics].fillna(0)

    # D. Feature Engineering
    all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
    all_data['TotalBath'] = (all_data['FullBath'] + (0.5 * all_data['HalfBath']) + 
                             all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath']))
    all_data['HasPool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    all_data['Has2ndFloor'] = all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    all_data['HasGarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

    # E. Skew Correction
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)
    skewness = skewed_feats[abs(skewed_feats) > 0.75]
    skewed_features = skewness.index
    all_data[skewed_features] = np.log1p(all_data[skewed_features])

    # F. Manual Ordinal Encoding
    ordinal_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
    ordinal_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 
                    'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']

    for col in ordinal_cols:
        all_data[col] = all_data[col].map(ordinal_map).fillna(0)

    # G. One-Hot Encoding
    all_data = pd.get_dummies(all_data)
    
    return all_data

def train():
    print("Loading data...")
    train_df = pd.read_csv('home-data-for-ml-course/train.csv', index_col='Id')
    test_df = pd.read_csv('home-data-for-ml-course/test.csv', index_col='Id')

    # Remove outliers
    train_df = train_df.drop(train_df[(train_df['GrLivArea'] > 4000) & (train_df['SalePrice'] < 300000)].index)

    y = np.log1p(train_df['SalePrice'])
    train_features = train_df.drop(['SalePrice'], axis=1)
    
    print("Preprocessing...")
    all_data = preprocess_data(train_features, test_df)
    
    X = all_data.iloc[:len(y), :]
    feature_columns = X.columns.tolist()

    # Scale data
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Define Models (hyperparams from notebook)
    xgboost = XGBRegressor(learning_rate=0.01, n_estimators=3460,
                           max_depth=3, min_child_weight=0,
                           gamma=0, subsample=0.7,
                           colsample_bytree=0.7,
                           objective='reg:squarederror', nthread=-1,
                           scale_pos_weight=1, seed=27,
                           reg_alpha=0.00006)

    gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                    max_depth=4, max_features='sqrt',
                                    min_samples_leaf=15, min_samples_split=10, 
                                    loss='huber', random_state=5)

    ridge = RidgeCV(alphas=[14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5])
    lasso = LassoCV(alphas=[0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1], 
                    max_iter=50000, cv=10)

    stack_gen = StackingRegressor(
        estimators=[
            ('xgb', xgboost), 
            ('gbr', gbr), 
            ('ridge', ridge), 
            ('lasso', lasso)
        ],
        final_estimator=RidgeCV(),
        n_jobs=-1
    )

    print("Training Stacking Ensemble (this may take a few minutes)...")
    stack_gen.fit(X_scaled, y)

    print("Saving model and artifacts...")
    os.makedirs('static', exist_ok=True)
    joblib.dump(stack_gen, 'static/model.joblib')
    joblib.dump(scaler, 'static/scaler.joblib')
    joblib.dump(feature_columns, 'static/feature_columns.joblib')
    
    # Save a reference row and unique values for categorical columns
    categorical_values = {col: train_features[col].unique().tolist() 
                          for col in train_features.select_dtypes(include=['object']).columns}
    joblib.dump(categorical_values, 'static/categorical_values.joblib')
    joblib.dump(train_features.iloc[:1], 'static/reference_row.joblib')
    
    print("Done!")

if __name__ == "__main__":
    train()
