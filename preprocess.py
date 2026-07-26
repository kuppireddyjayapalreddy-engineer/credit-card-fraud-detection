"""
preprocess.py
Handles all data cleaning and feature engineering steps:
  - Median imputation for missing values
  - StandardScaler on Amount + Time
  - SMOTE oversampling for class imbalance
Returns: X_train_bal, X_test, y_train_bal, y_test, imputer, scaler
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42
FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
TARGET_COL = "Class"


def load_from_db(db_path: str) -> pd.DataFrame:
    from sqlalchemy import create_engine
    engine = create_engine(f"sqlite:///{db_path}")
    df = pd.read_sql("SELECT * FROM transactions", engine)
    print(f"[preprocess] Loaded {len(df):,} rows from DB")
    return df


def clean_and_split(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy().reset_index(drop=True)
    X = X.reset_index(drop=True)

    # 1. Median imputation
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=FEATURE_COLS)

    # 2. Scale Amount and Time only (V1-V28 are already PCA-normalized)
    scaler = StandardScaler()
    X[["Amount", "Time"]] = scaler.fit_transform(X[["Amount", "Time"]])

    # 3. Train/test split (stratified to preserve fraud ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f"[preprocess] Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")
    print(f"[preprocess] Train fraud rate before SMOTE: {y_train.mean()*100:.2f}%")

    # 4. SMOTE on training data only (never on test)
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    X_train_bal = pd.DataFrame(X_train_bal, columns=FEATURE_COLS)
    y_train_bal = pd.Series(y_train_bal)

    print(f"[preprocess] After SMOTE — Train size: {len(X_train_bal):,}  |  Fraud: {y_train_bal.mean()*100:.1f}%")

    # Reset test indices
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train_bal, X_test, y_train_bal, y_test, imputer, scaler
