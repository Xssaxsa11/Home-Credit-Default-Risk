import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.preprocessing import prepare_features
from src.model import make_catboost
import json
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    recall_score,
    precision_score,
    f1_score
)

def main():
    Path('artifacts').mkdir(exist_ok=True)

    raw_df = pd.read_csv('data/application_train.csv')
    
    y = raw_df['TARGET']
    X = raw_df.drop(columns=['TARGET', 'SK_ID_CURR'])

    threshold = 0.40 #from experiments in notebook

    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, random_state=42 , test_size=0.3)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, stratify=y_val, random_state=42, test_size=0.5)

    feature_columns = X_train.columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'str']).columns.tolist()

    with open('artifacts/best_params.json', 'r', encoding='utf-8') as f:
        params = json.load(f)

    model = make_catboost(**params)

    X_train = prepare_features(X_train,feature_columns, categorical_cols)
    X_val = prepare_features(X_val, feature_columns, categorical_cols)
    X_test = prepare_features(X_test, feature_columns, categorical_cols)

    model.fit(X_train, y_train, cat_features=categorical_cols, eval_set=(X_val, y_val))
    model.save_model('artifacts/catboost.cbm')

    train_proba = model.predict_proba(X_train)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]
    test_preds = (test_proba >= threshold).astype(int)

    auc_score = roc_auc_score(y_test, test_proba)

    metrics = {
        'roc_auc_score': auc_score,
        'average_precision' : average_precision_score(y_test, test_proba),
        'recall': recall_score(y_test, test_preds),
        'precision' : precision_score(y_test, test_preds),
        'f1_score' : f1_score(y_test, test_preds),
        'trees': model.tree_count_,
        'threshold': threshold,
    }
    with open('artifacts/metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)

    print(
        f'Train roc_auc_score: {roc_auc_score(y_train, train_proba)}\n'
        f'Test roc_auc_score: {auc_score}'
        )

    schema = {
        'feature_columns': feature_columns,
        'categorical_columns': categorical_cols,
        'threshold' : threshold
    }

    with open('artifacts/schema.json', 'w', encoding='utf-8') as file:
        json.dump(schema, file , ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()