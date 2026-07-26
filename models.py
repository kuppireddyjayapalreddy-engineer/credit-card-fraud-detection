"""
models.py
Defines the two A/B test arms:
  - Model A (Control)   : Logistic Regression
  - Model B (Challenger): XGBoost

Both expose a common interface: train(), predict(), evaluate()
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier

RANDOM_STATE = 42


class BaseModel:
    name: str = "BaseModel"

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test, y_test) -> dict:
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)
        cm = confusion_matrix(y_test, y_pred)
        metrics = {
            "model":     self.name,
            "accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
            "tp": int(cm[1, 1]),
            "fp": int(cm[0, 1]),
            "tn": int(cm[0, 0]),
            "fn": int(cm[1, 0]),
        }
        return metrics

    def print_report(self, X_test, y_test):
        y_pred = self.predict(X_test)
        print(f"\n{'='*50}")
        print(f"  {self.name} — Classification Report")
        print('='*50)
        print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))


class LogisticRegressionModel(BaseModel):
    """Control arm — Logistic Regression baseline."""
    name = "Logistic Regression (Control)"

    def __init__(self):
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver="lbfgs",
        )


class XGBoostModel(BaseModel):
    """Challenger arm — XGBoost."""
    name = "XGBoost (Challenger)"

    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=1,        # SMOTE already balanced
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            verbosity=0,
        )
