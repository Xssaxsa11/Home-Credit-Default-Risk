import pandas as pd
from src.preprocessing import  prepare_features
import json
from catboost import CatBoostClassifier
import numpy as np

df = pd.read_csv('data/sample_inputs.csv')

with open('artifacts/schema.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

feature_cols = data['feature_columns'] 
categorical_cols = data['categorical_columns']

required_cols = set(feature_cols) | {'SK_ID_CURR'}
missing_cols = required_cols - set(df.columns)

if missing_cols:
    raise ValueError(f'Не хватает колонок: {missing_cols}')

X = prepare_features(df, feature_cols, categorical_cols)

model = CatBoostClassifier()
model.load_model('artifacts/catboost.cbm')

scores = model.predict_proba(X)[:, 1]

results = pd.DataFrame({
    'SK_ID_CURR' : df['SK_ID_CURR'].to_numpy(),
    'risk_scores' : scores
})

results = results.sort_values('risk_scores', ascending=False)

results.to_csv('reports/preditctions.csv', index=False)
