"""
integration_analytics.py
=========================
Part 3 – Analytics Integration
Covers:
  Business Question: Which customers have the highest lifetime value, and
  what behavioural signals (session engagement from HBase-style data) predict
  whether a user will become a high-value customer?

  Data sources involved:
    - User profiles   → users.json          (would sit in MongoDB)
    - Transactions    → transactions.json   (would sit in MongoDB)
    - Sessions        → sessions_*.json     (would sit in HBase / loaded into Spark)

  Processing:
    1. Load all three sources into Spark DataFrames (simulating a cross-store join)
    2. Compute transaction-based CLV features per user
    3. Compute session-based engagement features per user (total sessions,
       avg session duration, conversion rate, preferred device, referrer mix)
    4. Join both feature sets to build a unified user analytics profile
    5. Segment users by CLV tier and analyse behavioural differences

  This demonstrates the integration value: neither MongoDB nor HBase alone
  can answer this question — you need both purchase history and session
  behaviour, and Spark is the natural join layer.

Run with: python integration_analytics.py  (uses local PySpark)
"""

import os
import json
import glob
import pandas as pd
import numpy as np
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), ".")

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import DoubleType, IntegerType
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False


def get_spark():
    return (
        SparkSession.builder
        .appName("CLV_Integration")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK: Pure Pandas implementation (when PySpark unavailable)
# ─────────────────────────────────────────────────────────────────────────────

def run_pandas_integration():
    """
    Pandas-based CLV + engagement analysis.
    Equivalent to the Spark version but runs without a Spark cluster.
    Used as the primary execution path in this environment.
    """
    print("[Integration] Running with Pandas (equivalent to Spark cross-store join)\n")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("Loading data sources...")
    with open(os.path.join(DATA_DIR, "users.json")) as f:
        users = pd.DataFrame(json.load(f))

    with open(os.path.join(DATA_DIR, "transactions.json")) as f:
        txns = pd.DataFrame(json.load(f))

    session_files = sorted(glob.glob(os.path.join(DATA_DIR, "sessions_*.json")))
    sessions_list = []
    for sf in session_files:
        with open(sf) as f:
            sessions_list.extend(json.load(f))
    sessions = pd.DataFrame(sessions_list)

    print(f"  Users: {len(users):,}  |  Transactions: {len(txns):,}  |  Sessions: {len(sessions):,}")

    # ── 2. CLV features from transactions (MongoDB source) ───────────────────
    print("\nComputing CLV features from transactions (MongoDB layer)...")

    txns["timestamp"] = pd.to_datetime(txns["timestamp"], errors="coerce")
    txns["total"]     = pd.to_numeric(txns["total"], errors="coerce").fillna(0)
    txns["discount"]  = pd.to_numeric(txns["discount"], errors="coerce").fillna(0)
    txns_completed    = txns[txns["status"] == "completed"].copy()

    clv_features = txns_completed.groupby("user_id").agg(
        order_count    = ("transaction_id", "count"),
        total_spend    = ("total", "sum"),
        avg_order_val  = ("total", "mean"),
        total_discount = ("discount", "sum"),
        first_purchase = ("timestamp", "min"),
        last_purchase  = ("timestamp", "max")
    ).reset_index()

    clv_features["clv"] = clv_features["total_spend"].round(2)
    clv_features["lifespan_days"] = (
        (clv_features["last_purchase"] - clv_features["first_purchase"])
        .dt.days.fillna(0).astype(int)
    )
    clv_features["discount_rate_pct"] = (
        100 * clv_features["total_discount"] /
        (clv_features["total_spend"] + clv_features["total_discount"] + 0.01)
    ).round(2)

    # CLV tiers
    p33, p66 = clv_features["clv"].quantile([0.33, 0.66])
    clv_features["clv_tier"] = pd.cut(
        clv_features["clv"],
        bins=[-np.inf, p33, p66, np.inf],
        labels=["Low", "Mid", "High"]
    )
    print(f"  CLV tier thresholds: Low < ${p33:.2f} < Mid < ${p66:.2f} < High")

    # ── 3. Engagement features from sessions (HBase source) ──────────────────
    print("\nComputing engagement features from sessions (HBase layer)...")

    sessions["duration_seconds"] = pd.to_numeric(sessions["duration_seconds"], errors="coerce").fillna(0)

    engagement = sessions.groupby("user_id").agg(
        total_sessions    = ("session_id", "count"),
        avg_session_dur   = ("duration_seconds", "mean"),
        total_session_dur = ("duration_seconds", "sum"),
    ).reset_index()

    # Conversion rate per user (from sessions, not transactions)
    conv_counts = sessions.groupby(["user_id", "conversion_status"]).size().unstack(fill_value=0)
    for col in ["converted", "abandoned", "browsed"]:
        if col not in conv_counts.columns:
            conv_counts[col] = 0
    conv_counts["user_conversion_rate"] = (
        100 * conv_counts["converted"] /
        (conv_counts["converted"] + conv_counts["abandoned"] + conv_counts["browsed"] + 0.001)
    ).round(2)
    engagement = engagement.merge(
        conv_counts[["converted", "abandoned", "user_conversion_rate"]].reset_index(),
        on="user_id", how="left"
    )

    # Preferred device
    device_mode = sessions.groupby("user_id").apply(
        lambda x: x["device_profile"].apply(
            lambda d: d.get("type", "unknown") if isinstance(d, dict) else "unknown"
        ).mode().iloc[0] if len(x) > 0 else "unknown"
    ).reset_index()
    device_mode.columns = ["user_id", "preferred_device"]
    engagement = engagement.merge(device_mode, on="user_id", how="left")

    # Primary referrer
    ref_mode = sessions.groupby("user_id")["referrer"].agg(
        lambda x: x.mode().iloc[0] if len(x) > 0 else "unknown"
    ).reset_index()
    ref_mode.columns = ["user_id", "primary_referrer"]
    engagement = engagement.merge(ref_mode, on="user_id", how="left")

    engagement["avg_session_dur"] = engagement["avg_session_dur"].round(1)
    print(f"  Engagement profile computed for {len(engagement):,} users")

    # ── 4. Join: unified user analytics profile ───────────────────────────────
    print("\nJoining CLV + engagement (cross-store integration)...")

    users["registration_date"] = pd.to_datetime(users["registration_date"], errors="coerce")
    user_profiles = users[["user_id", "registration_date"]].copy()
    user_profiles["country"] = users["geo_data"].apply(
        lambda g: g.get("country", "Unknown") if isinstance(g, dict) else "Unknown"
    )

    unified = (
        clv_features
        .merge(engagement, on="user_id", how="left")
        .merge(user_profiles, on="user_id", how="left")
    )
    unified = unified.fillna({
        "total_sessions": 0,
        "avg_session_dur": 0,
        "user_conversion_rate": 0,
        "preferred_device": "unknown",
        "primary_referrer": "unknown"
    })

    print(f"  Unified profile rows: {len(unified):,}")

    # ── 5. CLV Segment Analysis ───────────────────────────────────────────────
    print("\n── CLV Segment Behavioural Analysis ─────────────────────────────────")
    segment_summary = unified.groupby("clv_tier", observed=True).agg(
        user_count            = ("user_id", "count"),
        avg_clv               = ("clv", "mean"),
        avg_order_count       = ("order_count", "mean"),
        avg_sessions          = ("total_sessions", "mean"),
        avg_session_dur_sec   = ("avg_session_dur", "mean"),
        avg_conversion_rate   = ("user_conversion_rate", "mean"),
        avg_discount_rate_pct = ("discount_rate_pct", "mean")
    ).round(2)

    print(segment_summary.to_string())

    # ── 6. Top CLV Users ──────────────────────────────────────────────────────
    print("\n── Top 10 Customers by Lifetime Value ────────────────────────────────")
    top_users = unified.nlargest(10, "clv")[[
        "user_id", "clv", "order_count", "total_sessions",
        "avg_session_dur", "user_conversion_rate", "preferred_device"
    ]]
    print(top_users.to_string(index=False))

    # ── 7. Referrer × CLV Analysis ────────────────────────────────────────────
    print("\n── Revenue Attribution by Referrer Channel ──────────────────────────")
    referrer_rev = unified.groupby("primary_referrer").agg(
        user_count   = ("user_id", "count"),
        total_rev    = ("clv", "sum"),
        avg_clv      = ("clv", "mean"),
        avg_conv_rate = ("user_conversion_rate", "mean")
    ).round(2).sort_values("total_rev", ascending=False)
    print(referrer_rev.to_string())

    # ── 8. Save results ───────────────────────────────────────────────────────
    unified.to_json(
        os.path.join(DATA_DIR, "clv_unified_profile.json"),
        orient="records", date_format="iso"
    )
    segment_summary.reset_index().to_json(
        os.path.join(DATA_DIR, "clv_segment_summary.json"),
        orient="records"
    )
    referrer_rev.reset_index().to_json(
        os.path.join(DATA_DIR, "referrer_revenue.json"),
        orient="records"
    )
    print(f"\n  ✓ Results saved to {DATA_DIR}")
    return unified, segment_summary, referrer_rev


# ─────────────────────────────────────────────────────────────────────────────
# SPARK VERSION  (identical logic, distributed)
# ─────────────────────────────────────────────────────────────────────────────

def run_spark_integration():
    """
    PySpark version of the same CLV integration pipeline.
    Identical business logic — scales to full 2M session, 500K transaction dataset.
    """
    spark = get_spark()
    spark.sparkContext.setLogLevel("ERROR")
    print("[Integration] Running with Apache Spark (distributed mode)\n")

    txn_df  = spark.read.json(os.path.join(DATA_DIR, "transactions.json"))
    users_df = spark.read.json(os.path.join(DATA_DIR, "users.json"))
    files   = sorted(glob.glob(os.path.join(DATA_DIR, "sessions_*.json")))
    sess_df = spark.read.json(files)

    txn_df.createOrReplaceTempView("transactions")
    users_df.createOrReplaceTempView("users")
    sess_df.createOrReplaceTempView("sessions")

    # CLV from transactions
    clv = spark.sql("""
        SELECT
            user_id,
            COUNT(*) AS order_count,
            ROUND(SUM(total), 2) AS clv,
            ROUND(AVG(total), 2) AS avg_order_value,
            MIN(to_date(timestamp)) AS first_purchase,
            MAX(to_date(timestamp)) AS last_purchase
        FROM transactions
        WHERE status = 'completed'
        GROUP BY user_id
    """)

    # Engagement from sessions
    engagement = spark.sql("""
        SELECT
            user_id,
            COUNT(*) AS total_sessions,
            ROUND(AVG(duration_seconds), 1) AS avg_session_dur,
            ROUND(100.0 * SUM(CASE WHEN conversion_status='converted' THEN 1 ELSE 0 END) / COUNT(*), 2)
                AS user_conversion_rate
        FROM sessions
        GROUP BY user_id
    """)

    # Join
    unified = (
        clv.join(engagement, on="user_id", how="left")
        .withColumn("clv_tier",
            F.when(F.col("clv") > 2000, "High")
             .when(F.col("clv") > 500,  "Mid")
             .otherwise("Low")
        )
    )

    print("CLV Segment Summary:")
    unified.groupBy("clv_tier").agg(
        F.count("user_id").alias("users"),
        F.round(F.avg("clv"), 2).alias("avg_clv"),
        F.round(F.avg("order_count"), 1).alias("avg_orders"),
        F.round(F.avg("total_sessions"), 1).alias("avg_sessions")
    ).orderBy("avg_clv", ascending=False).show()

    spark.stop()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Part 3: Integrated Analytics — Customer Lifetime Value")
    print("Cross-store: MongoDB (transactions) × HBase (sessions) × Spark")
    print("=" * 60)

    # Run pandas version (always available)
    unified, segments, referrer = run_pandas_integration()

    # Optionally run Spark version
    if SPARK_AVAILABLE:
        print("\n" + "="*60)
        print("Running Spark version (distributed equivalent)...")
        run_spark_integration()


if __name__ == "__main__":
    main()
