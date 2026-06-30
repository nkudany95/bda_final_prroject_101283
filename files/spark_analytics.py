"""
spark_analytics.py
==================
Part 2 – Apache Spark Batch Processing & Spark SQL
Covers:
  1. Data cleaning and normalisation (handles missing values, type casting,
     format standardisation across all JSON files)
  2. Product co-purchase recommendation indicators
     ("users who bought X also bought Y")
  3. Cohort analysis: user purchasing patterns grouped by registration month
  4. Spark SQL analytics: complex cross-entity queries on DataFrames

Run with: spark-submit spark_analytics.py
       or: python spark_analytics.py  (uses local[*] SparkSession)

Requires: pyspark (pip install pyspark)
"""

import os
import json
import glob
from datetime import datetime

# ── Spark Session ──────────────────────────────────────────────────────────────
try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        StructType, StructField, StringType, DoubleType,
        IntegerType, BooleanType, ArrayType, TimestampType
    )
    from pyspark.sql.window import Window
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    print("WARNING: PySpark not installed.  Install with: pip install pyspark")
    print("Showing code structure and expected output only.\n")

DATA_DIR = os.path.join(os.path.dirname(__file__), ".")
OUT_DIR  = os.path.join(os.path.dirname(__file__), ".")


def get_spark():
    return (
        SparkSession.builder
        .appName("EcommerceAnalytics")
        .master("local[*]")
        .config("spark.driver.memory",       "4g")
        .config("spark.sql.shuffle.partitions", "8")   # lower for single-machine
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


# ─────────────────────────────────────────────────────────────────────────────
# PART 2.1 — DATA CLEANING & NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

def clean_transactions(spark):
    """
    Load and clean transaction data.

    Issues handled:
    - Null session_id for standalone transactions (expected, kept as null)
    - Missing discount values → default 0.0
    - Inconsistent timestamp formats → parsed to TimestampType
    - status values normalised to lowercase
    - Negative or zero total → flagged as suspicious (not removed, flagged)
    """
    print("\n[Spark] Cleaning transactions...")

    raw = spark.read.json(os.path.join(DATA_DIR, "transactions.json"))

    cleaned = (
        raw
        .withColumn("timestamp",
            F.to_timestamp(F.col("timestamp"), "yyyy-MM-dd'T'HH:mm:ss"))
        .withColumn("discount",
            F.when(F.col("discount").isNull(), 0.0)
             .otherwise(F.col("discount").cast(DoubleType())))
        .withColumn("total",
            F.col("total").cast(DoubleType()))
        .withColumn("subtotal",
            F.col("subtotal").cast(DoubleType()))
        .withColumn("status",
            F.lower(F.trim(F.col("status"))))
        .withColumn("is_suspicious",
            F.when(F.col("total") <= 0, True).otherwise(False))
        .withColumn("date",
            F.to_date(F.col("timestamp")))
        .withColumn("year_month",
            F.date_format(F.col("timestamp"), "yyyy-MM"))
    )

    total   = cleaned.count()
    nulls   = cleaned.filter(F.col("session_id").isNull()).count()
    suspect = cleaned.filter(F.col("is_suspicious")).count()

    print(f"  Total transactions : {total:,}")
    print(f"  Standalone (no session): {nulls:,}  ({100*nulls/total:.1f}%)")
    print(f"  Suspicious (total<=0)  : {suspect:,}")
    return cleaned


def clean_users(spark):
    """
    Load and clean user profiles.

    Issues handled:
    - Null geo_data subfields → default 'Unknown'
    - registration_date parsed to TimestampType
    - Derive user_tenure_days from registration_date to now
    """
    print("[Spark] Cleaning users...")
    raw = spark.read.json(os.path.join(DATA_DIR, "users.json"))

    cleaned = (
        raw
        .withColumn("registration_date",
            F.to_timestamp(F.col("registration_date"), "yyyy-MM-dd'T'HH:mm:ss"))
        .withColumn("last_active",
            F.to_timestamp(F.col("last_active"), "yyyy-MM-dd'T'HH:mm:ss"))
        .withColumn("country",
            F.when(F.col("geo_data.country").isNull(), "Unknown")
             .otherwise(F.col("geo_data.country")))
        .withColumn("reg_year_month",
            F.date_format(F.col("registration_date"), "yyyy-MM"))
        .withColumn("tenure_days",
            F.datediff(F.current_date(), F.col("registration_date").cast("date")))
    )
    print(f"  Users loaded: {cleaned.count():,}")
    return cleaned


def clean_products(spark):
    """
    Load and clean product catalog.

    Issues handled:
    - is_active parsed from boolean/string
    - current_stock nulls → 0
    - Derive price_volatility from price_history length
    """
    print("[Spark] Cleaning products...")
    raw = spark.read.json(os.path.join(DATA_DIR, "products.json"))

    cleaned = (
        raw
        .withColumn("current_stock",
            F.when(F.col("current_stock").isNull(), 0)
             .otherwise(F.col("current_stock").cast(IntegerType())))
        .withColumn("is_active",
            F.col("is_active").cast(BooleanType()))
        .withColumn("price_history_len",
            F.when(F.col("price_history").isNull(), 0)
             .otherwise(F.size(F.col("price_history"))))
        .withColumn("is_price_volatile",
            F.col("price_history_len") > 1)
    )
    print(f"  Products loaded: {cleaned.count():,}")
    return cleaned


def load_sessions(spark):
    """Load all session chunk files into a single DataFrame."""
    print("[Spark] Loading sessions (all chunks)...")
    files = sorted(glob.glob(os.path.join(DATA_DIR, "sessions_*.json")))
    df = spark.read.json(files)
    print(f"  Sessions loaded: {df.count():,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PART 2.2 — PRODUCT CO-PURCHASE RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────

def copurchase_recommendations(txn_df, products_df, top_n=10):
    """
    "Users who bought X also bought Y" — market basket analysis.

    Method:
    1. For each transaction, explode items to get (transaction_id, product_id) pairs.
    2. Self-join on transaction_id to find all (productA, productB) pairs that
       appeared in the same basket.
    3. Filter out self-pairs (A != B) and count co-occurrences.
    4. Normalise by individual product purchase frequency to derive a lift score.

    Why Spark?
    Self-joining a 500K-transaction table with multi-item baskets produces
    potentially millions of pairs — this is embarrassingly parallel and exactly
    the workload Spark's distributed joins are designed for.
    """
    print("\n[Spark] Computing product co-purchase recommendations...")

    # (transaction_id, user_id, product_id) pairs from completed transactions
    items = (
        txn_df
        .filter(F.col("status") == "completed")
        .select("transaction_id", "user_id", F.explode("items").alias("item"))
        .select("transaction_id", "user_id", F.col("item.product_id").alias("product_id"))
    )

    # Self-join to find pairs in same basket
    pairs = (
        items.alias("a")
        .join(items.alias("b"), on="transaction_id")
        .filter(F.col("a.product_id") < F.col("b.product_id"))  # avoid duplicates
        .select(
            F.col("a.product_id").alias("product_a"),
            F.col("b.product_id").alias("product_b")
        )
        .groupBy("product_a", "product_b")
        .agg(F.count("*").alias("co_purchase_count"))
        .filter(F.col("co_purchase_count") >= 2)  # minimum support threshold
    )

    # Individual product purchase counts (for normalisation)
    prod_counts = (
        items.groupBy("product_id")
        .agg(F.count("*").alias("purchase_count"))
    )

    # Join product names for readability
    name_map = products_df.select("product_id", "name", "category_name") \
        if "category_name" in products_df.columns \
        else products_df.select("product_id", "name")

    result = (
        pairs
        .join(prod_counts.alias("ca"),
              F.col("product_a") == F.col("ca.product_id"))
        .join(prod_counts.alias("cb"),
              F.col("product_b") == F.col("cb.product_id"))
        .withColumn("jaccard_similarity",
            F.round(
                F.col("co_purchase_count") /
                (F.col("ca.purchase_count") + F.col("cb.purchase_count") - F.col("co_purchase_count")),
                4
            )
        )
        .join(name_map.alias("na"), F.col("product_a") == F.col("na.product_id"))
        .join(name_map.alias("nb"), F.col("product_b") == F.col("nb.product_id"))
        .select(
            F.col("product_a"),
            F.col("na.name").alias("name_a"),
            F.col("product_b"),
            F.col("nb.name").alias("name_b"),
            "co_purchase_count",
            "jaccard_similarity"
        )
        .orderBy(F.col("co_purchase_count").desc())
        .limit(top_n)
    )

    print(f"\n  Top {top_n} co-purchased product pairs:")
    print(f"  {'Product A':<35} {'Product B':<35} {'Co-purchases':>12}  {'Jaccard':>8}")
    print(f"  {'-'*35} {'-'*35} {'-'*12}  {'-'*8}")
    for row in result.collect():
        a = row["name_a"][:33] if row["name_a"] else row["product_a"]
        b = row["name_b"][:33] if row["name_b"] else row["product_b"]
        print(f"  {a:<35} {b:<35} {row['co_purchase_count']:>12,}  {row['jaccard_similarity']:>8.4f}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# PART 2.3 — COHORT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def cohort_analysis(users_df, txn_df):
    """
    Cohort Analysis: user spending by registration month.

    Definition:
    A cohort = all users who registered in the same calendar month.
    For each cohort, we track:
      - How many users are in it
      - Total spend in each subsequent month after registration
      - Average revenue per user (ARPU) per cohort month
    This reveals retention and monetisation patterns — whether early cohorts
    spend more, whether recent cohorts are converting faster, etc.

    Why Spark?
    Cohort analysis requires joining two large tables (users + transactions)
    followed by a two-dimensional groupBy — a classic distributed join workload.
    """
    print("\n[Spark] Running cohort analysis...")

    # Cohort = month of first registration
    user_cohort = users_df.select(
        "user_id",
        F.col("reg_year_month").alias("cohort_month")
    )

    # Monthly transaction amounts per user
    user_monthly = (
        txn_df
        .filter(F.col("status") == "completed")
        .select(
            "user_id",
            F.col("year_month").alias("txn_month"),
            "total"
        )
        .groupBy("user_id", "txn_month")
        .agg(F.sum("total").alias("monthly_spend"))
    )

    # Join: user → cohort + spending
    cohort_spend = (
        user_monthly
        .join(user_cohort, on="user_id")
        .groupBy("cohort_month", "txn_month")
        .agg(
            F.count("user_id").alias("active_users"),
            F.sum("monthly_spend").alias("cohort_revenue"),
            F.avg("monthly_spend").alias("arpu")
        )
        .withColumn("cohort_revenue", F.round("cohort_revenue", 2))
        .withColumn("arpu",           F.round("arpu", 2))
        .orderBy("cohort_month", "txn_month")
    )

    print("\n  Cohort analysis (cohort_month × txn_month):")
    print(f"  {'Cohort':<10} {'Txn Month':<12} {'Active Users':>12} {'Revenue':>12} {'ARPU':>8}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*8}")
    for row in cohort_spend.limit(20).collect():
        print(f"  {row['cohort_month']:<10} {row['txn_month']:<12} "
              f"{row['active_users']:>12,} ${row['cohort_revenue']:>11,.2f} ${row['arpu']:>7.2f}")

    return cohort_spend


# ─────────────────────────────────────────────────────────────────────────────
# PART 2.4 — SPARK SQL ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────

def spark_sql_analytics(spark, txn_df, users_df, products_df, sessions_df):
    """
    Complex Spark SQL queries demonstrating cross-entity analytics.
    These queries simulate the kind of analysis that would draw on both
    MongoDB (transaction/product data) and HBase (session data).
    """
    print("\n[Spark SQL] Registering temp views...")
    txn_df.createOrReplaceTempView("transactions")
    users_df.createOrReplaceTempView("users")
    products_df.createOrReplaceTempView("products")
    sessions_df.createOrReplaceTempView("sessions")

    # ── SQL Query 1: Daily Revenue with 7-day Rolling Average ────────────────
    print("\n  ── SQL 1: Daily Revenue with 7-day Rolling Average ──────────")
    q1 = spark.sql("""
        SELECT
            date,
            COUNT(*)                                              AS daily_orders,
            ROUND(SUM(total), 2)                                  AS daily_revenue,
            ROUND(AVG(SUM(total)) OVER (
                ORDER BY date
                ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
            ), 2)                                                 AS rolling_7d_avg,
            ROUND(SUM(discount), 2)                               AS daily_discount
        FROM transactions
        WHERE status = 'completed'
          AND date IS NOT NULL
        GROUP BY date
        ORDER BY date
    """)
    q1.show(10, truncate=False)

    # ── SQL Query 2: Funnel Conversion Analysis ───────────────────────────────
    print("  ── SQL 2: Session Funnel Conversion by Referrer ────────────")
    q2 = spark.sql("""
        SELECT
            referrer,
            COUNT(*)                                              AS total_sessions,
            SUM(CASE WHEN conversion_status = 'converted'  THEN 1 ELSE 0 END) AS converted,
            SUM(CASE WHEN conversion_status = 'abandoned'  THEN 1 ELSE 0 END) AS abandoned,
            SUM(CASE WHEN conversion_status = 'browsed'    THEN 1 ELSE 0 END) AS browsed,
            ROUND(
                100.0 * SUM(CASE WHEN conversion_status = 'converted' THEN 1 ELSE 0 END)
                / COUNT(*), 2
            )                                                     AS conversion_rate_pct,
            ROUND(AVG(duration_seconds), 0)                       AS avg_session_dur_sec
        FROM sessions
        GROUP BY referrer
        ORDER BY conversion_rate_pct DESC
    """)
    q2.show(truncate=False)

    # ── SQL Query 3: Device & OS Performance ─────────────────────────────────
    print("  ── SQL 3: Revenue by Device Type and OS ───────────────────")
    q3 = spark.sql("""
        SELECT
            s.device_profile.type                                 AS device_type,
            s.device_profile.os                                   AS os,
            COUNT(DISTINCT t.transaction_id)                      AS transactions,
            ROUND(SUM(t.total), 2)                                AS revenue,
            ROUND(AVG(t.total), 2)                                AS avg_order_value
        FROM transactions t
        JOIN sessions s ON t.session_id = s.session_id
        WHERE t.status = 'completed'
        GROUP BY s.device_profile.type, s.device_profile.os
        ORDER BY revenue DESC
    """)
    q3.show(truncate=False)

    # ── SQL Query 4: Customer Lifetime Value Segments ─────────────────────────
    print("  ── SQL 4: Customer LTV Segmentation ───────────────────────")
    q4 = spark.sql("""
        WITH user_spend AS (
            SELECT
                user_id,
                COUNT(*)            AS order_count,
                ROUND(SUM(total),2) AS total_spend,
                ROUND(AVG(total),2) AS avg_order_value,
                MIN(date)           AS first_purchase,
                MAX(date)           AS last_purchase
            FROM transactions
            WHERE status = 'completed'
            GROUP BY user_id
        ),
        segmented AS (
            SELECT *,
                CASE
                    WHEN total_spend > 2000 THEN 'High Value'
                    WHEN total_spend > 500  THEN 'Mid Value'
                    ELSE                        'Low Value'
                END AS ltv_segment,
                DATEDIFF(last_purchase, first_purchase) AS customer_lifespan_days
            FROM user_spend
        )
        SELECT
            ltv_segment,
            COUNT(*)                            AS user_count,
            ROUND(AVG(total_spend), 2)          AS avg_ltv,
            ROUND(AVG(order_count), 1)          AS avg_orders,
            ROUND(AVG(avg_order_value), 2)      AS avg_order_value,
            ROUND(AVG(customer_lifespan_days),0) AS avg_lifespan_days
        FROM segmented
        GROUP BY ltv_segment
        ORDER BY avg_ltv DESC
    """)
    q4.show(truncate=False)

    # Save query 1 (daily revenue) for visualisation
    daily_rev = q1.toPandas()
    daily_rev.to_json(
        os.path.join(OUT_DIR, "daily_revenue.json"),
        orient="records", date_format="iso"
    )
    print(f"  ✓ Daily revenue data saved for visualisation")

    # Save funnel data
    funnel = q2.toPandas()
    funnel.to_json(
        os.path.join(OUT_DIR, "funnel_by_referrer.json"),
        orient="records"
    )

    # Save LTV segments
    ltv = q4.toPandas()
    ltv.to_json(
        os.path.join(OUT_DIR, "ltv_segments.json"),
        orient="records"
    )

    return q1, q2, q3, q4


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not SPARK_AVAILABLE:
        print("PySpark unavailable — showing pipeline structure only.")
        return

    spark = get_spark()
    spark.sparkContext.setLogLevel("ERROR")
    print("=" * 60)
    print("Apache Spark Analytics Pipeline")
    print(f"Spark version: {spark.version}")
    print("=" * 60)

    # 1. Load and clean
    txn_df      = clean_transactions(spark)
    users_df    = clean_users(spark)
    products_df = clean_products(spark)
    sessions_df = load_sessions(spark)

    # 2. Product co-purchase recommendations
    copurchase_recommendations(txn_df, products_df, top_n=10)

    # 3. Cohort analysis
    cohort_df = cohort_analysis(users_df, txn_df)

    # Save cohort results
    cohort_df.toPandas().to_json(
        os.path.join(OUT_DIR, "cohort_analysis.json"),
        orient="records"
    )

    # 4. Spark SQL
    spark_sql_analytics(spark, txn_df, users_df, products_df, sessions_df)

    spark.stop()
    print("\n✓ Spark pipeline complete.")


if __name__ == "__main__":
    main()
