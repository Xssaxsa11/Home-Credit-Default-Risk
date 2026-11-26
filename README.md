# Home Credit Default Risk

An ML project for credit default risk prediction using CatBoost, scikit-learn and Optuna.
The model estimates the risk of default on credit applications, with a focus on reducing false negatives and prioritizing applications for manual review.

## Dataset
The dataset is from the [Home Credit Default Risk competition on Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk).
The project uses `application_train.csv`, containing 307,511 applications with an imbalanced binary target (`TARGET`).

## Methodology

### Feature Engineering

After experimenting with the existing features, three additional features were selected: income-to-credit, goods-price-to-credit and annuity-to-income ratios. The preprocessing also handles missing categorical values and anomalous employment data.

For Logistic Regression, a separate scikit-learn preprocessing pipeline handles numerical and categorical features using imputation, scaling and target encoding.

### Data Splitting

The data was split into training, validation and test sets (70/15/15) using stratified `train_test_split` from `sklearn.model_selection`. The validation set was used for early stopping, hyperparameter optimization and threshold selection. The test set was reserved for final evaluation.

### Hyperparameter Optimization

An Optuna study with 10 trials was used to tune CatBoost's depth, learning rate and L2 regularization, maximizing validation ROC-AUC. Other parameters were kept fixed, including balanced class weights to account for class imbalance. A broader search could be explored in future experiments.

## Model Performance

Comparison of models based on validation ROC-AUC:

| Model | Validation ROC-AUC |
| --- | ---: |
| DummyClassifier | 0.500000 |
| Logistic Regression | 0.745312 |
| CatBoost | 0.761181 |
| CatBoost (Optuna) | **0.761554** |

Final model metrics on the test set (CatBoost with Optuna, threshold = 0.40):

| Metric | Test result |
|---|---:|
| ROC-AUC | 0.7642 |
| Average Precision | 0.2484 |
| Recall | 80.96% |
| Precision | 14.03% |
| F1-score | 0.2391 |

## Decision Strategy

The classification threshold of 0.40 was selected on the validation set to achieve at least 80% recall while maximizing precision among the tested thresholds that met this requirement. False negatives are particularly important because they represent potential defaults that the model fails to identify. The resulting low precision is a trade-off for higher recall.

The model is intended to assist rather than fully automate credit risk assessment. It flags 10% of applications (rounded up) for additional manual review, prioritizing applications predicted as low risk that are closest to the decision threshold. If there are not enough such applications, the remaining review slots are filled with high-risk predictions closest to the threshold.

On the test set, 224 out of 709 false negatives (31.59%) were flagged for manual review. Assuming perfect manual review, recall could potentially increase from 80.96% to 86.98%.

This is an educational project, not a production-ready lending system. The risk scores are not calibrated default probabilities.

## Project Structure

```text
HCDR-public/
├── artifacts/
│   ├── best_params.json
│   ├── catboost.cbm
│   ├── metrics.json
│   └── schema.json
├── data/
│   └── sample_inputs.csv
├── notebooks/
│   └── experiments.ipynb
├── reports/
│   └── predictions.csv
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
├── tests/
│   ├── test_preprocessing.py
│   └── test_predict.py
├── .gitignore
├── requirements.txt
└── README.md
```
## Installation and Usage

The project was developed using Python 3.14. Run the following commands from the project root.

Install the required libraries:

```bash
pip install -r requirements.txt
```

To retrain the model, download `application_train.csv` from Kaggle and place it in the `data/` directory:

```bash
python -m src.train
```

To generate predictions for `sample_inputs.csv` using the pre-trained model (no full dataset required):

```bash
python -m src.predict
```

The results are saved to `reports/predictions.csv`. Each prediction includes the application ID, risk score, predicted class and manual review flag.

The model experiments are available in [notebooks/experiments.ipynb](notebooks/experiments.ipynb).

## Testing

The project includes 11 automated tests covering feature engineering, input validation, prediction logic and manual review prioritization.

```bash
python -m pytest -v
```

## Future Improvements

- FastAPI integration and Docker containerization.
- Additional feature engineering.
- A broader hyperparameter search.
- Probability calibration.




