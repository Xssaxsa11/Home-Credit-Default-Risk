import pytest
from src.preprocessing import prepare_features
import pandas as pd
import numpy as np

@pytest.fixture
def sample_df():
    df = pd.DataFrame({
        'AMT_CREDIT' : [12, 50],
        'AMT_INCOME_TOTAL':[50, 100],
        'AMT_GOODS_PRICE': [50,200],
        'AMT_ANNUITY': [70, 1000],
        'DAYS_EMPLOYED': [604, 32],
        'Categorical_col': ['Working', np.nan]
    })

    feature_cols = df.columns
    cat_cols = ['Categorical_col']

    return df, feature_cols , cat_cols

def test_feature_creation(sample_df):

    result = prepare_features(*sample_df)

    assert {'Income_div_credit', 'Price_div_amount', 'Annuity_div_income'}.issubset(result.columns)

def test_feature_values(sample_df):

    result = prepare_features(*sample_df)

    assert result['Income_div_credit'].tolist() == pytest.approx([50/12, 100/50])
    assert result['Price_div_amount'].tolist() == pytest.approx([50/12, 200/50])
    assert result['Annuity_div_income'].tolist() == pytest.approx([70/50, 1000/100])

def test_zero_division(sample_df):
    sample_df[0].loc[1, ['AMT_CREDIT', 'AMT_INCOME_TOTAL']] = [0, 0]

    result = prepare_features(*sample_df)

    cols = ['Income_div_credit', 'Price_div_amount', 'Annuity_div_income']

    assert pd.isna(result.loc[1, cols]).all()

def test_employment_anomaly(sample_df):
    sample_df[0].loc[1,'DAYS_EMPLOYED'] = 365243

    result = prepare_features(*sample_df)

    assert pd.isna(result.loc[1, 'DAYS_EMPLOYED'])

def test_cat_missing_values(sample_df):
    result = prepare_features(*sample_df)

    assert result.loc[1, 'Categorical_col'] == 'missing'