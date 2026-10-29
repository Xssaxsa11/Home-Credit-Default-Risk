import numpy as np

def prepare_features(df, feature_columns, categorical_columns):
    X = df[feature_columns].copy()
    # feature engineering
    credit = X['AMT_CREDIT'].replace(0, np.nan)
    income = X['AMT_INCOME_TOTAL'].replace(0, np.nan)
    
    X['Income_div_credit'] = income / credit
    X['Price_div_amount'] = X['AMT_GOODS_PRICE'] / credit
    X['Annuity_div_income'] = X['AMT_ANNUITY'] / income
    #ended feature enginnering
    
    X['DAYS_EMPLOYED'] = X['DAYS_EMPLOYED'].replace(365243, np.nan)
    X[categorical_columns] = X[categorical_columns].fillna('missing').astype('str')

    

    return X
