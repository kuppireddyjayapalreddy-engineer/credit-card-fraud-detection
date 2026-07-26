"""
tests/test_models.py
Unit tests for model training, prediction, and evaluation.
Run with: pytest tests/test_models.py -v
"""
import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import LogisticRegressionModel, XGBoostModel


@pytest.fixture
def small_dataset():
    """Tiny balanced synthetic dataset for fast unit tests."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((200, 30)),
                     columns=[f"V{i}" for i in range(1, 29)] + ["Time", "Amount"])
    y = pd.Series([0] * 100 + [1] * 100)
    return X, y


class TestLogisticRegression:
    def test_train_predict_shape(self, small_dataset):
        X, y = small_dataset
        model = LogisticRegressionModel()
        model.train(X, y)
        preds = model.predict(X)
        assert preds.shape == (200,), "Prediction shape mismatch"

    def test_predict_binary_output(self, small_dataset):
        X, y = small_dataset
        model = LogisticRegressionModel().train(X, y)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1}), "Predictions must be 0 or 1"

    def test_predict_proba_range(self, small_dataset):
        X, y = small_dataset
        model = LogisticRegressionModel().train(X, y)
        probs = model.predict_proba(X)
        assert probs.min() >= 0.0 and probs.max() <= 1.0, "Probabilities out of [0,1]"

    def test_evaluate_returns_required_keys(self, small_dataset):
        X, y = small_dataset
        model = LogisticRegressionModel().train(X, y)
        metrics = model.evaluate(X, y)
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            assert key in metrics, f"Missing metric: {key}"

    def test_accuracy_above_chance(self, small_dataset):
        X, y = small_dataset
        model = LogisticRegressionModel().train(X, y)
        metrics = model.evaluate(X, y)
        assert metrics["accuracy"] > 0.5, "Model worse than random chance"


class TestXGBoost:
    def test_train_predict_shape(self, small_dataset):
        X, y = small_dataset
        model = XGBoostModel()
        model.train(X, y)
        preds = model.predict(X)
        assert preds.shape == (200,)

    def test_predict_binary_output(self, small_dataset):
        X, y = small_dataset
        model = XGBoostModel().train(X, y)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_range(self, small_dataset):
        X, y = small_dataset
        model = XGBoostModel().train(X, y)
        probs = model.predict_proba(X)
        assert probs.min() >= 0.0 and probs.max() <= 1.0

    def test_evaluate_returns_required_keys(self, small_dataset):
        X, y = small_dataset
        model = XGBoostModel().train(X, y)
        metrics = model.evaluate(X, y)
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            assert key in metrics

    def test_xgb_beats_chance(self, small_dataset):
        X, y = small_dataset
        model = XGBoostModel().train(X, y)
        metrics = model.evaluate(X, y)
        assert metrics["f1"] > 0.5

    def test_confusion_matrix_fields(self, small_dataset):
        X, y = small_dataset
        model = XGBoostModel().train(X, y)
        metrics = model.evaluate(X, y)
        assert metrics["tp"] + metrics["fn"] == y.sum(), "TP+FN must equal actual positives"
