"""
tests/test_preprocess.py
Tests for the preprocessing pipeline.
Run with: pytest tests/test_preprocess.py -v
"""
import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.generate_data import generate_dataset
from src.preprocess import clean_and_split, FEATURE_COLS


@pytest.fixture(scope="module")
def preprocessed():
    df = generate_dataset()
    return clean_and_split(df)


class TestCleanAndSplit:
    def test_returns_six_items(self, preprocessed):
        assert len(preprocessed) == 6

    def test_no_nulls_in_X_train(self, preprocessed):
        X_train, X_test, y_train, y_test, imputer, scaler = preprocessed
        assert X_train.isna().sum().sum() == 0, "Training set has missing values"

    def test_no_nulls_in_X_test(self, preprocessed):
        X_train, X_test, y_train, y_test, imputer, scaler = preprocessed
        assert X_test.isna().sum().sum() == 0, "Test set has missing values"

    def test_smote_balanced_training(self, preprocessed):
        X_train, X_test, y_train, y_test, imputer, scaler = preprocessed
        # y_train here is y_train_bal (after SMOTE)
        fraud_rate = y_train.mean()
        assert 0.4 <= fraud_rate <= 0.6, f"SMOTE didn't balance classes: {fraud_rate:.3f}"

    def test_test_set_preserves_original_fraud_rate(self, preprocessed):
        X_train, X_test, y_train, y_test, imputer, scaler = preprocessed
        fraud_rate = y_test.mean()
        assert fraud_rate < 0.01, f"Test fraud rate unexpectedly high: {fraud_rate:.4f}"

    def test_feature_columns_preserved(self, preprocessed):
        X_train, X_test, y_train, y_test, imputer, scaler = preprocessed
        assert list(X_train.columns) == FEATURE_COLS
        assert list(X_test.columns) == FEATURE_COLS

    def test_train_larger_than_test(self, preprocessed):
        X_train, X_test, y_train, y_test, imputer, scaler = preprocessed
        assert len(X_train) > len(X_test), "Train set should be larger than test"

    def test_test_size_approx_20_percent(self, preprocessed):
        X_train, X_test, y_train, y_test, imputer, scaler = preprocessed
        assert len(X_test) > 50_000, "Test set unexpectedly small"

    def test_amount_std_reasonable_after_scaling(self, preprocessed):
        X_train, X_test, y_train, y_test, imputer, scaler = preprocessed
        # After scaling std should be close to 1
        std = X_test["Amount"].std()
        assert 0.5 <= std <= 2.0, f"Amount std unexpected after scaling: {std:.3f}"

    def test_time_std_reasonable_after_scaling(self, preprocessed):
        X_train, X_test, y_train, y_test, imputer, scaler = preprocessed
        std = X_test["Time"].std()
        assert 0.5 <= std <= 2.0, f"Time std unexpected after scaling: {std:.3f}"
