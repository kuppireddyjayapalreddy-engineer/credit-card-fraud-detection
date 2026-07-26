"""
generate_data.py
Simulates the Kaggle Credit Card Fraud dataset (284,807 rows, 0.17% fraud).
Produces a CSV and loads it into SQLite for the pipeline.
"""
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import os

RANDOM_STATE = 42
N_NORMAL = 28_4315
N_FRAUD = 492
DB_PATH = os.path.join(os.path.dirname(__file__), "../db/fraud.db")
CSV_PATH = os.path.join(os.path.dirname(__file__), "../data/transactions.csv")


def generate_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)

    # --- Normal transactions ---
    normal = pd.DataFrame(
        rng.standard_normal((N_NORMAL, 28)),
        columns=[f"V{i}" for i in range(1, 29)],
    )
    normal["Time"] = rng.uniform(0, 172_800, N_NORMAL)
    normal["Amount"] = rng.lognormal(3.5, 1.5, N_NORMAL).clip(0.01, 25_000)
    normal["Class"] = 0

    # --- Fraudulent transactions (shifted distributions) ---
    fraud = pd.DataFrame(
        rng.standard_normal((N_FRAUD, 28)) * 1.8 + rng.choice([-2, 2], (N_FRAUD, 28)),
        columns=[f"V{i}" for i in range(1, 29)],
    )
    fraud["Time"] = rng.choice(
        np.concatenate([rng.uniform(0, 21_600, N_FRAUD // 2),          # off-hours
                        rng.uniform(162_000, 172_800, N_FRAUD // 2)]),
        N_FRAUD, replace=False,
    )
    fraud["Amount"] = rng.lognormal(2.0, 0.8, N_FRAUD).clip(0.5, 500)  # smaller amounts
    fraud["Class"] = 1

    df = pd.concat([normal, fraud], ignore_index=True).sample(
        frac=1, random_state=RANDOM_STATE
    ).reset_index(drop=True)

    # Introduce ~1% missing values in 5 random columns to simulate real data quality issues
    for col in rng.choice(df.columns[:-1], 5, replace=False):
        mask = rng.random(len(df)) < 0.01
        df.loc[mask, col] = np.nan

    return df


def save_to_csv(df: pd.DataFrame):
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"[generate_data] Saved {len(df):,} rows to {CSV_PATH}")


def save_to_db(df: pd.DataFrame):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    engine = create_engine(f"sqlite:///{DB_PATH}")
    df.to_sql("transactions", engine, if_exists="replace", index=False)
    print(f"[generate_data] Saved to SQLite: {DB_PATH}")


if __name__ == "__main__":
    df = generate_dataset()
    save_to_csv(df)
    save_to_db(df)
    print(f"  Total rows : {len(df):,}")
    print(f"  Fraud rows : {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")
