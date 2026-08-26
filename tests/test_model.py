import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.train import build_model


def train_small_model():
    """Train the project model on a small, predictable dataset."""
    X, y = make_classification(
        n_samples=200,
        n_features=8,
        n_informative=6,
        n_redundant=0,
        class_sep=2.0,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    parameters = {
            'C': 1.0,
            'max_iter': 1000,
            'class_weight': 'balanced',
            'solver': 'liblinear'
    }

    model = build_model(
        parameters=parameters,
        random_state=42,
    )
    model.fit(X_train, y_train)

    return model, X_test, y_test


def test_model_prediction_type_and_shape():
    model, X_test, _ = train_small_model()
    predictions = model.predict(X_test)

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (len(X_test),)


def test_model_meets_minimum_performance():
    model, X_test, y_test = train_small_model()
    predictions = model.predict(X_test)

    score = f1_score(y_test, predictions)

    minimum_f1 = 0.35

    assert score >= minimum_f1