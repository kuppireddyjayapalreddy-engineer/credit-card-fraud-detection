"""
db_logger.py
Logs every prediction and A/B test result to SQLite.
Tables:
  - predictions       : per-transaction model scores
  - ab_test_results   : aggregated metrics per model run
  - ab_test_winner    : the declared winner with rationale
"""
import datetime
import os
import pandas as pd
from sqlalchemy import create_engine, text

DB_PATH = os.path.join(os.path.dirname(__file__), "../db/fraud.db")


def get_engine(db_path: str = DB_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return create_engine(f"sqlite:///{db_path}")


def init_tables(db_path: str = DB_PATH):
    engine = get_engine(db_path)
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS predictions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id      TEXT,
                model_name  TEXT,
                transaction_idx INTEGER,
                y_true      INTEGER,
                y_pred      INTEGER,
                y_prob      REAL,
                ts          TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ab_test_results (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id     TEXT,
                model_name TEXT,
                accuracy   REAL,
                precision  REAL,
                recall     REAL,
                f1         REAL,
                roc_auc    REAL,
                tp INTEGER, fp INTEGER, tn INTEGER, fn INTEGER,
                ts         TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ab_test_winner (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id     TEXT,
                winner     TEXT,
                rationale  TEXT,
                ts         TEXT
            )
        """))
        conn.commit()
    print("[db_logger] Tables initialised")


def log_predictions(run_id: str, model_name: str,
                    y_true, y_pred, y_prob,
                    db_path: str = DB_PATH):
    engine = get_engine(db_path)
    ts = datetime.datetime.utcnow().isoformat()
    rows = [
        {
            "run_id": run_id,
            "model_name": model_name,
            "transaction_idx": int(i),
            "y_true": int(yt),
            "y_pred": int(yp),
            "y_prob": float(ypr),
            "ts": ts,
        }
        for i, (yt, yp, ypr) in enumerate(zip(y_true, y_pred, y_prob))
    ]
    pd.DataFrame(rows).to_sql("predictions", engine, if_exists="append", index=False)
    print(f"[db_logger] Logged {len(rows):,} predictions for '{model_name}'")


def log_ab_result(run_id: str, metrics: dict, db_path: str = DB_PATH):
    engine = get_engine(db_path)
    row = {**metrics, "run_id": run_id, "ts": datetime.datetime.utcnow().isoformat()}
    row.pop("model", None)
    row["model_name"] = metrics["model"]
    pd.DataFrame([row]).to_sql("ab_test_results", engine, if_exists="append", index=False)
    print(f"[db_logger] Logged A/B result for '{metrics['model']}'")


def log_winner(run_id: str, winner: str, rationale: str, db_path: str = DB_PATH):
    engine = get_engine(db_path)
    row = {"run_id": run_id, "winner": winner,
           "rationale": rationale, "ts": datetime.datetime.utcnow().isoformat()}
    pd.DataFrame([row]).to_sql("ab_test_winner", engine, if_exists="append", index=False)
    print(f"[db_logger] Winner logged: {winner}")


def fetch_ab_results(db_path: str = DB_PATH) -> pd.DataFrame:
    engine = get_engine(db_path)
    return pd.read_sql("SELECT * FROM ab_test_results ORDER BY ts DESC", engine)


def fetch_winner(db_path: str = DB_PATH) -> pd.DataFrame:
    engine = get_engine(db_path)
    return pd.read_sql("SELECT * FROM ab_test_winner ORDER BY ts DESC LIMIT 1", engine)
