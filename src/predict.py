import pandas as pd
from src.preprocessing import  prepare_features
import json
from catboost import CatBoostClassifier
import numpy as np
from pathlib import Path
import os

def load_artefacts(art_dir):
    model = CatBoostClassifier()
    model.load_model(os.path.join(art_dir, 'catboost.cbm'))

    with open(os.path.join(art_dir, 'schema.json'), 'r', encoding='utf-8') as file:
            schema = json.load(file)

    return model, schema

def predict(df, model, schema):
    feature_cols = schema['feature_columns'] 
    categorical_cols = schema['categorical_columns']
    threshold = schema['threshold']
    
    required_cols = set(feature_cols) | {'SK_ID_CURR'}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
            raise ValueError(f'Missed required cols: {missing_cols}')
    
    X = prepare_features(df, feature_cols, categorical_cols)

    scores = model.predict_proba(X)[:, 1]
    
    predictions = (scores >=threshold).astype(int)

    results = pd.DataFrame({
            'SK_ID_CURR' : df['SK_ID_CURR'].to_numpy(),
            'risk_scores' : scores,
            'predictions' : predictions
        })
    results['distance'] = abs(results['risk_scores'] - threshold)
    n_review = int(np.ceil(len(results) * 0.10))

    review_idx = results.sort_values(by=['predictions', 'distance']).iloc[:n_review].index
        
    results['need_review'] = False
    results.loc[review_idx, 'need_review'] = True
    
    results = results.sort_values('risk_scores', ascending=False)
    results = results.drop(columns='distance')
    return results


def main():
    model, schema = load_artefacts('artifacts')

    df = pd.read_csv('data/sample_inputs.csv')

    predictions = predict(df, model, schema)

    path = Path('reports')
    path.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(path / 'predictions.csv', index=False)

    print(f'Processed application: {len(predictions)}')
    print('Result was saved in reports/predictions.csv')
    
    

if __name__ == '__main__':
      main()

