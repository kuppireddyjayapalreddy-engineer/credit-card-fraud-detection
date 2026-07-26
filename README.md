# Credit Card Fraud Detection — Part II
## End-to-End ML Pipeline with A/B Testing

### Project Structure
```
fraud_part2/
├── src/
│   ├── generate_data.py    # Simulates 284,807 transactions (0.17% fraud) → SQLite + CSV
│   ├── preprocess.py       # Median imputation → StandardScaler → SMOTE
│   ├── models.py           # LogisticRegression (Control) + XGBoost (Challenger)
│   └── db_logger.py        # Logs all predictions & A/B results to SQLite
├── tests/
│   ├── test_data.py        # 16 data quality tests (schema, imbalance, distributions)
│   ├── test_models.py      # 11 unit tests for LR and XGBoost
│   └── test_preprocess.py  # 10 preprocessing pipeline tests
├── ab_test.py              # Main pipeline: train → simulate → evaluate → log winner
├── spark_pipeline.py       # STRETCH GOAL: PySpark feature engineering + EDA at scale
├── db/fraud.db             # SQLite: transactions + predictions + A/B results
└── mlruns/                 # MLflow experiment tracking
```

### How to Run

**Install dependencies:**
```bash
pip install scikit-learn xgboost imbalanced-learn pandas numpy sqlalchemy pytest mlflow pyspark
```

**Run all tests (37 tests):**
```bash
pytest tests/ -v
```

**Run the full A/B test pipeline:**
```bash
python ab_test.py
```

**Run the PySpark pipeline (stretch goal):**
```bash
python spark_pipeline.py
```

### What Each Step Does

1. **Data Generation** — Simulates the Kaggle credit card fraud dataset with 284,807 transactions,
   492 fraud cases (0.17%), missing values, and realistic fraud patterns (low amounts, off-hours).
   Saved to SQLite `transactions` table.

2. **Preprocessing** — Median imputation → StandardScaler on Amount/Time →
   Stratified 80/20 train-test split → SMOTE balances training data to 50/50.

3. **A/B Traffic Split** — Test set is split 50/50, simulating production routing
   where half of incoming transactions go to each model.

4. **Model Training**
   - **Control (A):** Logistic Regression with class_weight='balanced'
   - **Challenger (B):** XGBoost (200 trees, depth=6)
   Both trained on identical SMOTE-balanced training data.

5. **Evaluation & Logging** — Per-transaction predictions logged to `predictions` table.
   Aggregated metrics logged to `ab_test_results`. Winner logged to `ab_test_winner`.
   All runs tracked in MLflow.

6. **Winner Declaration** — Primary metric: F1 score (best for fraud imbalance).
   Secondary: ROC-AUC.

### Database Tables (SQLite)
| Table | Contents |
|---|---|
| `transactions` | Raw input data (284,807 rows) |
| `predictions` | Per-transaction scores from both models |
| `ab_test_results` | Aggregated metrics per model per run |
| `ab_test_winner` | Declared winner + rationale per run |

### A/B Test Results
| Metric | Control (LR) | Challenger (XGB) |
|---|---|---|
| Accuracy | 0.66 | 1.00 |
| Precision | 0.004 | 1.00 |
| Recall | 0.85 | 1.00 |
| F1 | 0.008 | 1.00 |
| ROC-AUC | 0.82 | 1.00 |

**Winner: XGBoost** — Promotes to production.

### Stretch Goal: PySpark
`spark_pipeline.py` loads the full dataset via Spark, runs EDA (class distribution,
amount stats by class), builds a Spark ML Pipeline with VectorAssembler + StandardScaler,
and computes hourly fraud rate — replicating a Databricks CE workflow locally.
