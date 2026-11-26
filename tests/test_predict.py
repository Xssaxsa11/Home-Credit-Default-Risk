import pytest
from src.predict import load_artefacts,predict
import pandas as pd
import numpy as np

@pytest.fixture
def artifacts():
    art_dir = 'artifacts'
    
    model, schema = load_artefacts(art_dir)

    df = pd.read_csv('data/sample_inputs.csv')

    return df,model,schema

def test_predicted_cols(artifacts):
    results = predict(*artifacts)

    assert {'SK_ID_CURR', 'risk_scores' , 'predictions', 'need_review'}.issubset(results.columns)

def test_predictions_count(artifacts):
    results = predict(*artifacts)

    assert len(artifacts[0]) == len(results)

def test_review_proportion(artifacts):
    results = predict(*artifacts)

    assert len(results[results['need_review'] == True]) == np.ceil(len(artifacts[0]) / 10.0)

def test_prediction_threshold(artifacts):
    results = predict(*artifacts)
    threshold = artifacts[2]['threshold']

    assert (results[results['predictions'] == 0]['risk_scores'] < threshold).all()
    assert (results[results['predictions'] == 1]['risk_scores'] >= threshold).all()

def test_missing_columns(artifacts):
    df, model, schema = artifacts

    df = df.drop(columns=['AMT_CREDIT'])

    with pytest.raises(ValueError):
        predict(df, model, schema)

def test_review_priority(artifacts):
    results = predict(*artifacts)
    threshold = artifacts[2]['threshold']
    n_review = int(np.ceil(len(results) * 0.10))

    results['distance'] = abs(results['risk_scores'] - threshold)

    expected_idx = results.nsmallest(
        n_review, ['predictions', 'distance']
    ).index

    actual_idx = results[results['need_review']].index

    assert set(actual_idx) == set(expected_idx)