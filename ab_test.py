"""
ab_test.py
Simulates production A/B testing:
  - 50/50 traffic split between Control (LR) and Challenger (XGBoost)
  - Both models trained on same balanced training data
  - Evaluated on the same held-out test set
  - Winner declared based on F1 score (best for imbalanced fraud detection)
  - All results logged to SQLite + MLflow
"""
import os, uuid, sys
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost

sys.path.insert(0, os.path.dirname(__file__))

from src.generate_data import generate_dataset, save_to_db, DB_PATH as DEFAULT_DB
from src.preprocess import load_from_db, clean_and_split
from src.models import LogisticRegressionModel, XGBoostModel
from src.db_logger import init_tables, log_predictions, log_ab_result, log_winner

DB_PATH    = os.path.join(os.path.dirname(__file__), "db", "fraud.db")
MLFLOW_DIR = os.path.join(os.path.dirname(__file__), "mlruns")


def run_ab_test():
    run_id = str(uuid.uuid4())[:8]
    print(f"\n{'='*60}")
    print(f"  CREDIT CARD FRAUD DETECTION — A/B TEST  [run: {run_id}]")
    print(f"{'='*60}\n")

    # ── 1. Data ──────────────────────────────────────────────────
    print("► Step 1: Generating dataset...")
    df = generate_dataset()
    import src.generate_data as gd
    orig_path = gd.DB_PATH
    gd.DB_PATH = DB_PATH
    gd.save_to_db(df)
    gd.DB_PATH = orig_path

    init_tables(DB_PATH)
    df = load_from_db(DB_PATH)

    # ── 2. Preprocess ────────────────────────────────────────────
    print("\n► Step 2: Preprocessing...")
    X_train, X_test, y_train, y_test, imputer, scaler = clean_and_split(df)

    # ── 3. A/B Traffic Split Simulation ─────────────────────────
    print("\n► Step 3: Simulating 50/50 production traffic split...")
    rng = np.random.default_rng(42)
    idx = np.arange(len(X_test))
    rng.shuffle(idx)
    half = len(idx) // 2
    idx_a, idx_b = idx[:half], idx[half:]

    X_test_a = X_test.iloc[idx_a].reset_index(drop=True)
    y_test_a = y_test.iloc[idx_a].reset_index(drop=True)
    X_test_b = X_test.iloc[idx_b].reset_index(drop=True)
    y_test_b = y_test.iloc[idx_b].reset_index(drop=True)
    print(f"   Traffic A (Control)   : {len(X_test_a):,} transactions")
    print(f"   Traffic B (Challenger): {len(X_test_b):,} transactions")

    # ── 4. Train & Evaluate Models ───────────────────────────────
    print("\n► Step 4: Training models...")
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DIR}/mlflow.db")
    mlflow.set_experiment("CreditCardFraud_AB_Test")

    results = {}

    # --- Model A: Logistic Regression ---
    with mlflow.start_run(run_name=f"Control_LR_{run_id}"):
        model_a = LogisticRegressionModel()
        print("   Training Logistic Regression (Control)...")
        model_a.train(X_train, y_train)
        metrics_a = model_a.evaluate(X_test_a, y_test_a)
        model_a.print_report(X_test_a, y_test_a)
        mlflow.log_params({"model": "LogisticRegression", "solver": "lbfgs", "max_iter": 1000})
        mlflow.log_metrics({k: v for k, v in metrics_a.items() if isinstance(v, float)})
        mlflow.sklearn.log_model(model_a.model, "model")
        log_predictions(run_id, metrics_a["model"],
                        y_test_a, model_a.predict(X_test_a), model_a.predict_proba(X_test_a), DB_PATH)
        log_ab_result(run_id, metrics_a, DB_PATH)
        results["A"] = metrics_a

    # --- Model B: XGBoost ---
    with mlflow.start_run(run_name=f"Challenger_XGB_{run_id}"):
        model_b = XGBoostModel()
        print("   Training XGBoost (Challenger)...")
        model_b.train(X_train, y_train)
        metrics_b = model_b.evaluate(X_test_b, y_test_b)
        model_b.print_report(X_test_b, y_test_b)
        mlflow.log_params({"model": "XGBoost", "n_estimators": 200, "max_depth": 6, "lr": 0.1})
        mlflow.log_metrics({k: v for k, v in metrics_b.items() if isinstance(v, float)})
        mlflow.xgboost.log_model(model_b.model, "model")
        log_predictions(run_id, metrics_b["model"],
                        y_test_b, model_b.predict(X_test_b), model_b.predict_proba(X_test_b), DB_PATH)
        log_ab_result(run_id, metrics_b, DB_PATH)
        results["B"] = metrics_b

    # ── 5. A/B Results Table ─────────────────────────────────────
    print(f"\n► Step 5: A/B Test Results")
    print(f"\n{'─'*60}")
    print(f"{'Metric':<15} {'Control (LR)':>20} {'Challenger (XGB)':>20}")
    print(f"{'─'*60}")
    for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        a_val, b_val = results["A"][m], results["B"][m]
        mark = "◄ BETTER" if b_val > a_val else ""
        print(f"{m:<15} {a_val:>20.4f} {b_val:>20.4f}  {mark}")
    print(f"{'─'*60}")

    # ── 6. Declare Winner ────────────────────────────────────────
    if results["B"]["f1"] > results["A"]["f1"]:
        winner = results["B"]["model"]
        rationale = (
            f"XGBoost F1={results['B']['f1']:.4f} > LR F1={results['A']['f1']:.4f}. "
            f"ROC-AUC: {results['B']['roc_auc']:.4f} vs {results['A']['roc_auc']:.4f}. "
            "Challenger wins — promote to production."
        )
    else:
        winner = results["A"]["model"]
        rationale = (
            f"LR F1={results['A']['f1']:.4f} >= XGBoost F1={results['B']['f1']:.4f}. "
            "Control holds — keep current model."
        )

    log_winner(run_id, winner, rationale, DB_PATH)
    print(f"\n🏆 WINNER: {winner}")
    print(f"   Rationale: {rationale}")
    print(f"\n   Run ID: {run_id}")
    print(f"   Results saved → SQLite ({DB_PATH})")
    print(f"   MLflow logs  → {MLFLOW_DIR}")
    return results


if __name__ == "__main__":
    run_ab_test()
