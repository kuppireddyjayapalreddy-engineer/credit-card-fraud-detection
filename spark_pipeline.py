"""
spark_pipeline.py  — STRETCH GOAL
Reads transactions from CSV using PySpark, applies feature engineering,
and computes fraud statistics at scale (simulates Databricks CE workflow).
"""
import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

CSV_PATH = os.path.join(os.path.dirname(__file__), "data/transactions.csv")


def build_spark_session():
    return (
        SparkSession.builder
        .appName("CreditCardFraudDetection")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def run_spark_pipeline():
    spark = build_spark_session()
    spark.sparkContext.setLogLevel("ERROR")

    print("\n[Spark] Reading transactions CSV...")
    df = spark.read.csv(CSV_PATH, header=True, inferSchema=True)
    print(f"[Spark] Total rows: {df.count():,}")

    # ── Basic EDA at scale ──────────────────────────────────────
    print("\n[Spark] Class distribution:")
    df.groupBy("Class").count().show()

    print("[Spark] Amount stats by class:")
    df.groupBy("Class").agg(
        F.round(F.mean("Amount"), 2).alias("avg_amount"),
        F.round(F.stddev("Amount"), 2).alias("std_amount"),
        F.round(F.max("Amount"), 2).alias("max_amount"),
    ).show()

    # ── Feature Engineering ─────────────────────────────────────
    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]

    # Cast all to double (safety)
    for c in feature_cols:
        df = df.withColumn(c, F.col(c).cast(DoubleType()))

    # Drop nulls (Spark ML doesn't handle NaN by default)
    df_clean = df.dropna(subset=feature_cols)
    print(f"[Spark] After dropna: {df_clean.count():,} rows")

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features",
                             withMean=True, withStd=True)

    pipeline = Pipeline(stages=[assembler, scaler])
    model = pipeline.fit(df_clean)
    df_scaled = model.transform(df_clean)

    print("[Spark] Sample scaled feature vectors (first 3 rows):")
    df_scaled.select("Class", "features").show(3, truncate=80)

    # ── Hourly fraud rate (simulated monitoring metric) ──────────
    df_with_hour = df_clean.withColumn("hour", (F.col("Time") / 3600).cast("int") % 24)
    print("[Spark] Fraud count by simulated hour-of-day (top 5):")
    (
        df_with_hour
        .filter(F.col("Class") == 1)
        .groupBy("hour")
        .count()
        .orderBy(F.desc("count"))
        .show(5)
    )

    spark.stop()
    print("[Spark] Pipeline complete.")


if __name__ == "__main__":
    run_spark_pipeline()
