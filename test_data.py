"""
tests/test_data.py
Data quality / validation tests (Great Expectations style, but pure pytest).
Checks the generated dataset meets the expectations set in Part I EDA.
Run with: pytest tests/test_data.py -v
"""
import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.generate_data import generate_dataset

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]


@pytest.fixture(scope="module")
def df():
    return generate_dataset()


# ── Schema Tests ─────────────────────────────────────────────────

class TestSchema:
    def test_expected_columns_present(self, df):
        expected = set(FEATURE_COLS + ["Class"])
        assert expected.issubset(set(df.columns)), "Missing expected columns"

    def test_row_count_approximately_correct(self, df):
        assert 280_000 <= len(df) <= 290_000, f"Unexpected row count: {len(df)}"

    def test_class_column_is_binary(self, df):
        assert set(df["Class"].unique()).issubset({0, 1}), "Class must be 0 or 1"

    def test_amount_column_is_non_negative(self, df):
        assert (df["Amount"] >= 0).all(), "Amount contains negative values"

    def test_time_column_is_non_negative(self, df):
        assert (df["Time"] >= 0).all(), "Time contains negative values"


# ── Class Imbalance Tests ────────────────────────────────────────

class TestClassImbalance:
    def test_fraud_rate_below_1_percent(self, df):
        fraud_rate = df["Class"].mean()
        assert fraud_rate < 0.01, f"Fraud rate {fraud_rate:.4f} exceeds 1%"

    def test_fraud_rate_above_0_point_1_percent(self, df):
        fraud_rate = df["Class"].mean()
        assert fraud_rate > 0.001, f"Fraud rate {fraud_rate:.4f} suspiciously low"

    def test_fraud_count_at_least_400(self, df):
        assert df["Class"].sum() >= 400, "Too few fraud samples"


# ── Missing Value Tests ──────────────────────────────────────────

class TestMissingValues:
    def test_class_column_has_no_nulls(self, df):
        assert df["Class"].isna().sum() == 0, "Target column has missing values"

    def test_missing_rate_per_column_below_5_percent(self, df):
        rates = df[FEATURE_COLS].isna().mean()
        bad_cols = rates[rates > 0.05].to_dict()
        assert not bad_cols, f"Columns with >5% missing: {bad_cols}"

    def test_total_missing_rate_below_2_percent(self, df):
        total_cells = df[FEATURE_COLS].size
        total_missing = df[FEATURE_COLS].isna().sum().sum()
        rate = total_missing / total_cells
        assert rate < 0.02, f"Overall missing rate too high: {rate:.4f}"


# ── Distribution Tests ───────────────────────────────────────────

class TestDistributions:
    def test_fraud_mean_amount_lower_than_normal(self, df):
        fraud_mean = df[df["Class"] == 1]["Amount"].mean()
        normal_mean = df[df["Class"] == 0]["Amount"].mean()
        assert fraud_mean < normal_mean, (
            f"Expected fraud amount ({fraud_mean:.2f}) < normal ({normal_mean:.2f})"
        )

    def test_v_features_roughly_standard_normal(self, df):
        """V1-V28 come from PCA so should be roughly N(0,1)."""
        for col in [f"V{i}" for i in range(1, 29)]:
            col_mean = df[col].dropna().mean()
            assert abs(col_mean) < 1.0, f"{col} mean too far from 0: {col_mean:.3f}"

    def test_amount_has_positive_variance(self, df):
        assert df["Amount"].std() > 0, "Amount has zero variance"

    def test_time_spans_at_least_one_day(self, df):
        assert df["Time"].max() > 86_400, "Time doesn't span a full day"


# ── Duplicate Test ───────────────────────────────────────────────

class TestDuplicates:
    def test_no_fully_duplicate_rows(self, df):
        n_dupes = df.duplicated().sum()
        assert n_dupes == 0, f"Found {n_dupes} duplicate rows"
