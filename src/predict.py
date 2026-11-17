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
            data = json.load(file)

    return model, data        

def predict(df, model, schema, threshold = 0.5):
    feature_cols = schema['feature_columns'] 
    categorical_cols = schema['categorical_columns']
     
    required_cols = set(feature_cols) | {'SK_ID_CURR'}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
            raise ValueError(f'Не хватает колонок: {missing_cols}')
    
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
    # review_idx = results.nsmallest(n_review, 'distance').index
    
    results['need_review'] = False
    results.loc[review_idx, 'need_review'] = True
    
    results = results.sort_values('risk_scores', ascending=False)
    results = results.drop(columns='distance')
    return results


def main():
    model, schema = load_artefacts('artifacts')

    df = pd.read_csv('data/sample_inputs.csv')

    predictions = predict(df, model, schema, 0.41)

    path = Path('reports')
    path.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(path / 'predictions.csv', index=False)

    print(f'Обработано заявок: {len(predictions)}')
    print('Результат сохранён в reports/predictions.csv')
    
    

if __name__ == '__main__':
      main()

