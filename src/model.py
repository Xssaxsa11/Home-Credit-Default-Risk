from catboost import CatBoostClassifier

def make_catboost():
    model = CatBoostClassifier(
        iterations=300,
        depth=4,
        learning_rate=0.05,
        loss_function='Logloss',
        auto_class_weights='Balanced',
        random_seed=42,
        verbose=50
    )

    return model
