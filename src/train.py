import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from src.preprocessing import prepare_features
from src.model import make_catboost
import json
from pathlib import Path

def main():
    Path('artifacts').mkdir(exist_ok=True)

    raw_df = pd.read_csv('data/application_train.csv')
    
    y = raw_df['TARGET']
    X = raw_df.drop(columns=['TARGET', 'SK_ID_CURR'])

    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

    feature_columns = X_train.columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'str']).columns.tolist()

    model = make_catboost()

    X_train = prepare_features(X_train,feature_columns, categorical_cols)
    X_val = prepare_features(X_val, feature_columns, categorical_cols)

    model.fit(X_train, y_train, cat_features=categorical_cols)
    model.save_model('artifacts/catboost.cbm')

    train_proba = model.predict_proba(X_train)[:, 1]
    val_proba = model.predict_proba(X_val)[:, 1]

    print(
        f'Train roc_auc_score: {roc_auc_score(y_train, train_proba)}\n'
        f'Val roc_auc_score: {roc_auc_score(y_val, val_proba)}'
        )

    schema = {
        'feature_columns': feature_columns,
        'categorical_columns': categorical_cols
    }

    with open('artifacts/schema.json', 'w', encoding='utf-8') as file:
        json.dump(schema, file , ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()