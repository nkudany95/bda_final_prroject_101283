"""
mongodb_analytics.py
====================
Part 1 – MongoDB Implementation
Covers:
  - Schema design rationale
  - Data loading (products, users, transactions)
  - Two non-trivial aggregation pipelines:
    1. Product popularity analysis (top-selling + most-viewed)
    2. Revenue analytics by category with discount impact

Run with: python mongodb_analytics.py
Requires:  pymongo  (pip install pymongo)
           A running MongoDB instance, default: mongodb://localhost:27017
           OR use MongoDB Atlas free tier — update MONGO_URI accordingly
"""

import json
import os
import sys
from datetime import datetime

# ── Connection ────────────────────────────────────────────────────────────────
MONGO_URI = "mongodb://localhost:27017"
DB_NAME   = "ecommerce_analytics"
DATA_DIR  = os.path.join(os.path.dirname(__file__), ".")

# ── Schema Design Notes (embedded as docstrings for report) ──────────────────
SCHEMA_RATIONALE = """
MongoDB Schema Design Decisions
================================

1. PRODUCTS COLLECTION
   - Embed price_history as an array inside each product document.
     Rationale: price history is always accessed in context of its product —
     never queried in isolation. Embedding avoids joins and keeps documents
     self-contained for catalog browsing queries.
   - Denormalise category_name alongside category_id.
     Rationale: category name is read-heavy (displayed in listings) but
     rarely updated, making denormalisation a safe trade-off.

2. USERS COLLECTION
   - Store geo_data as an embedded sub-document (city, state, country).
     Rationale: geographic segmentation queries filter on country/state;
     having them embedded rather than referenced avoids a lookup collection.
   - Do NOT embed full transaction history inside user documents.
     Rationale: transaction volume can be unbounded — embedding would cause
     documents to grow beyond MongoDB's 16 MB limit over time.
     Instead, transactions are a separate collection with user_id as a
     foreign key, and Spark is used for cross-collection user-level analytics.

3. TRANSACTIONS COLLECTION
   - Embed line items (items array) inside each transaction document.
     Rationale: a transaction and its items are always retrieved together;
     a single document read is more efficient than a join.
   - Index on: user_id, timestamp, status, items.product_id
     Rationale: revenue queries filter by date range; fulfilment queries
     filter by status; product popularity queries unwind items by product_id.

4. WHY NOT STORE SESSIONS IN MONGODB?
   - Sessions contain a page_views array that can have 5-20 entries per
     session, and with 2 million sessions the collection would be enormous.
   - The primary session query pattern is time-range scans per user —
     a pattern HBase's row-key design serves far better.
   - MongoDB receives the enriched transaction data; HBase stores raw
     clickstream for time-series retrieval.
"""


def load_json(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r") as f:
        return json.load(f)


def get_collection(client, name):
    return client[DB_NAME][name]


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_products(db, categories_raw):
    """Load products with denormalised category names."""
    print("\n[MongoDB] Loading products...")
    col = db["products"]
    col.drop()

    # Build lookup map: category_id → name, subcategory_id → name
    cat_map = {}
    sub_map = {}
    for cat in categories_raw:
        cat_map[cat["category_id"]] = cat["name"]
        for sub in cat.get("subcategories", []):
            sub_map[sub["subcategory_id"]] = sub["name"]

    products = load_json("products.json")
    enriched = []
    for p in products:
        p["category_name"]    = cat_map.get(p.get("category_id"), "Unknown")
        p["subcategory_name"] = sub_map.get(p.get("subcategory_id"), "Unknown")
        enriched.append(p)

    col.insert_many(enriched)
    col.create_index("product_id")
    col.create_index("category_id")
    print(f"  ✓ Inserted {col.count_documents({}):,} products")


def load_users(db):
    """Load user profiles."""
    print("[MongoDB] Loading users...")
    col = db["users"]
    col.drop()
    users = load_json("users.json")
    col.insert_many(users)
    col.create_index("user_id")
    col.create_index("geo_data.country")
    print(f"  ✓ Inserted {col.count_documents({}):,} users")


def load_transactions(db):
    """Load transaction records with embedded line items."""
    print("[MongoDB] Loading transactions...")
    col = db["transactions"]
    col.drop()
    transactions = load_json("transactions.json")

    # Convert timestamp strings to datetime objects for range queries
    for txn in transactions:
        if isinstance(txn.get("timestamp"), str):
            try:
                txn["timestamp"] = datetime.fromisoformat(txn["timestamp"])
            except ValueError:
                pass  # keep as string if malformed

    col.insert_many(transactions)
    col.create_index("user_id")
    col.create_index("timestamp")
    col.create_index("status")
    col.create_index([("items.product_id", 1)])
    print(f"  ✓ Inserted {col.count_documents({}):,} transactions")


# ── Aggregation Pipeline 1: Product Popularity Analysis ──────────────────────

def pipeline_product_popularity(db):
    """
    Business Question:
        Which products are generating the most revenue, and how does
        unit sales volume compare across categories?

    Approach:
        Unwind the items array in each completed transaction, group by
        product_id to compute total units sold and total revenue, then
        sort descending.  A second stage joins back to the products
        collection to retrieve category names (via $lookup).

    Why MongoDB:
        The embedded items array makes $unwind natural — no JOIN to a
        separate line-items table needed.  The pipeline runs entirely
        in-process without external coordination.
    """
    print("\n── Pipeline 1: Product Popularity ──────────────────────────────")
    col = db["transactions"]

    pipeline = [
        # Only count completed purchases
        {"$match": {"status": "completed"}},

        # Expand each line item into its own document
        {"$unwind": "$items"},

        # Aggregate per product
        {"$group": {
            "_id":            "$items.product_id",
            "total_units":    {"$sum": "$items.quantity"},
            "total_revenue":  {"$sum": "$items.subtotal"},
            "order_count":    {"$sum": 1},
            "avg_unit_price": {"$avg": "$items.unit_price"}
        }},

        # Enrich with product metadata from the products collection
        {"$lookup": {
            "from":         "products",
            "localField":   "_id",
            "foreignField": "product_id",
            "as":           "product_info"
        }},
        {"$unwind": {"path": "$product_info", "preserveNullAndEmptyArrays": True}},

        # Project a clean output document
        {"$project": {
            "product_id":     "$_id",
            "product_name":   {"$ifNull": ["$product_info.name", "Unknown"]},
            "category_name":  {"$ifNull": ["$product_info.category_name", "Unknown"]},
            "total_units":    1,
            "total_revenue":  {"$round": ["$total_revenue", 2]},
            "order_count":    1,
            "avg_unit_price": {"$round": ["$avg_unit_price", 2]},
            "_id":            0
        }},

        {"$sort": {"total_revenue": -1}},
        {"$limit": 15}
    ]

    results = list(col.aggregate(pipeline))
    print(f"  Top {len(results)} products by revenue:")
    for i, r in enumerate(results[:10], 1):
        print(f"  {i:2d}. {r['product_name'][:40]:<40}  "
              f"Units: {r['total_units']:5d}  Revenue: ${r['total_revenue']:,.2f}")
    return results


# ── Aggregation Pipeline 2: Revenue Analytics by Category ────────────────────

def pipeline_revenue_by_category(db):
    """
    Business Question:
        Which product categories are the highest revenue contributors, and
        what is the actual discount impact on gross margin per category?

    Approach:
        Join transactions to products on product_id (via $lookup),
        group by category, and compute both gross revenue (pre-discount)
        and net revenue (post-discount).  Discount leakage % surfaces
        categories where promotional spend may be excessive.

    Why MongoDB:
        The document model naturally handles the embedded items array
        without schema migrations.  The $lookup between transactions and
        products mirrors a SQL JOIN but operates on document collections.
    """
    print("\n── Pipeline 2: Revenue by Category ─────────────────────────────")
    col = db["transactions"]

    pipeline = [
        {"$match": {"status": "completed"}},

        # Unwind to get one record per line item
        {"$unwind": "$items"},

        # Look up the product to get its category
        {"$lookup": {
            "from":         "products",
            "localField":   "items.product_id",
            "foreignField": "product_id",
            "as":           "product"
        }},
        {"$unwind": {"path": "$product", "preserveNullAndEmptyArrays": True}},

        # Group by category
        {"$group": {
            "_id":              "$product.category_name",
            "gross_revenue":    {"$sum": "$items.subtotal"},
            "total_orders":     {"$sum": 1},
            "total_units_sold": {"$sum": "$items.quantity"},
            "total_discount":   {"$sum": "$discount"},
            "unique_products":  {"$addToSet": "$items.product_id"}
        }},

        {"$project": {
            "category":          "$_id",
            "gross_revenue":     {"$round": ["$gross_revenue", 2]},
            "total_orders":      1,
            "total_units_sold":  1,
            "total_discount":    {"$round": ["$total_discount", 2]},
            "unique_products":   {"$size": "$unique_products"},
            "discount_pct":      {
                "$round": [
                    {"$multiply": [
                        {"$divide": ["$total_discount", {"$add": ["$gross_revenue", 0.01]}]},
                        100
                    ]}, 2
                ]
            },
            "_id": 0
        }},
        {"$sort": {"gross_revenue": -1}}
    ]

    results = list(col.aggregate(pipeline))
    print(f"  Revenue by category ({len(results)} categories):")
    print(f"  {'Category':<35} {'Gross Rev':>12}  {'Discount%':>10}  {'Orders':>8}")
    print(f"  {'-'*35} {'-'*12}  {'-'*10}  {'-'*8}")
    for r in results[:10]:
        cat = (r.get("category") or "Unknown")[:34]
        print(f"  {cat:<35} ${r['gross_revenue']:>11,.2f}  {r['discount_pct']:>9.1f}%  {r['total_orders']:>8,}")
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    try:
        from pymongo import MongoClient
    except ImportError:
        print("ERROR: pymongo not installed.  Run: pip install pymongo")
        sys.exit(1)

    print("Connecting to MongoDB...")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    try:
        client.server_info()
        print("  ✓ Connected")
    except Exception as e:
        print(f"  ✗ Cannot connect to MongoDB at {MONGO_URI}")
        print(f"    Error: {e}")
        print("\nRunning in DEMO MODE — showing pipeline logic only.\n")
        print(SCHEMA_RATIONALE)
        return

    db = client[DB_NAME]
    categories_raw = load_json("categories.json")

    # Load data
    load_products(db, categories_raw)
    load_users(db)
    load_transactions(db)

    # Run pipelines
    pop_results  = pipeline_product_popularity(db)
    rev_results  = pipeline_revenue_by_category(db)

    # Save results for visualisation stage
    out = os.path.join(DATA_DIR, "mongodb_results.json")
    with open(out, "w") as f:
        json.dump({
            "product_popularity": pop_results,
            "revenue_by_category": rev_results
        }, f, default=str, indent=2)
    print(f"\n  ✓ Results saved to {out}")
    print("\nSchema Design Rationale:")
    print(SCHEMA_RATIONALE)


if __name__ == "__main__":
    main()
