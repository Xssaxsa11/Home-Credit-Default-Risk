from catboost import CatBoostClassifier

def make_catboost(**params):
    model = CatBoostClassifier(
        **params,
        iterations=1000,
        loss_function='Logloss',
        auto_class_weights='Balanced',
        eval_metric='AUC', 
        early_stopping_rounds=50,
        use_best_model=True,
        random_seed=42,
        verbose=50
    )

    return model
