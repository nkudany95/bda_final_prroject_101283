"""
hbase_schema.py
===============
Part 1 – HBase Implementation
Covers:
  - HBase table design with column family rationale
  - Row key strategy for time-series data
  - Shell commands for table creation (DDL)
  - Python simulation of HBase query patterns using a dict-based store
  - Real HBase client code using happybase (when HBase is available)

The simulation is intentional: HBase requires a running Hadoop/HBase cluster
which is impractical in every dev environment.  The simulation produces
identical output to real HBase queries, and all row key / column family
decisions are production-grade.

Run with: python hbase_schema.py
"""

import json
import os
import sys
from datetime import datetime, timedelta
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), ".")


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA DESIGN  (embedded for report)
# ─────────────────────────────────────────────────────────────────────────────

HBASE_SCHEMA_RATIONALE = """
HBase Schema Design Decisions
==============================

TABLE 1: user_sessions
─────────────────────
Row Key:  {user_id}#{reverse_timestamp}
  e.g.    user_000042#9999999999999 - 1710253042000  → sorts most-recent first

Why reverse timestamp?
  HBase rows are stored in lexicographic order of the row key.
  For the query "give me the last 10 sessions for user X", we scan from
  user_X#0000000000000 forward — but that hits the oldest sessions first.
  Reversing the timestamp (Long.MAX_VALUE - epochMs) ensures the most recent
  sessions appear first in a forward scan, avoiding an expensive reverse scan.
  This is a standard HBase pattern for time-series data.

Column Families:
  meta   → session metadata (immutable after session close)
           meta:start_time, meta:end_time, meta:duration_seconds,
           meta:device_type, meta:device_os, meta:browser,
           meta:referrer, meta:conversion_status
  geo    → geographic context (low cardinality, rarely updated)
           geo:city, geo:state, geo:country, geo:ip_address
  cart   → cart state at end of session (sparse for browsed sessions)
           cart:product_ids (JSON array), cart:total_value

Why NOT embed page_views in HBase?
  Each session has 5-20 page views.  Storing each page view as a separate
  column qualifier (pv:0001, pv:0002 …) would work but creates very wide rows.
  Better pattern: store page views in a separate table (see TABLE 2) and
  co-locate rows with the same user_id prefix using region pre-splitting.

TABLE 2: product_performance
─────────────────────────────
Row Key:  {product_id}#{YYYYMMDD}
  e.g.    prod_00123#20250315

Column Families:
  views   → daily view aggregates
            views:count, views:unique_users, views:avg_duration_sec
  cart    → daily cart interaction aggregates
            cart:add_count, cart:remove_count, cart:total_value_added
  sales   → daily sales aggregates (populated from transactions)
            sales:units_sold, sales:revenue, sales:order_count

Why this row key?
  Querying a product's performance over a date range (e.g., last 30 days)
  becomes a single row scan: Scan(startRow='prod_00123#20250215',
  stopRow='prod_00123#20250315').  This is O(date range) not O(table size).

TABLE 3: page_events  (high-volume raw events)
───────────────────────────────────────────────
Row Key:  {user_id}#{session_id}#{seq_num}
Column Families:
  ev  → event:page_type, event:product_id, event:category_id,
        event:view_duration, event:timestamp

This table is write-optimised (append-only events) and never updated.
VERSIONS=1 keeps storage compact (no old value retention needed).

Comparison: MongoDB vs HBase
==============================
| Concern              | MongoDB                      | HBase                        |
|----------------------|------------------------------|------------------------------|
| Data access pattern  | Rich document queries        | Key-range scans              |
| Schema flexibility   | Schema-less, nested docs     | Fixed column families        |
| Transactions         | Multi-doc ACID (4.0+)        | Single-row atomicity only    |
| Best for             | Product catalog, orders      | Clickstream, time-series     |
| Query language       | MQL / aggregation pipelines  | Scan + Filter API            |
| Horizontal scale     | Sharding (manual config)     | Auto-sharding via regions    |
"""

# ─────────────────────────────────────────────────────────────────────────────
# HBASE SHELL DDL COMMANDS
# ─────────────────────────────────────────────────────────────────────────────

HBASE_SHELL_COMMANDS = """
# ============================================================
# HBase Shell — Table Creation DDL
# Run these in the HBase shell (hbase shell) or via Docker:
# docker exec -it hbase hbase shell
# ============================================================

# TABLE 1: User sessions (time-series, reverse timestamp row key)
create 'user_sessions',
  {NAME => 'meta',  VERSIONS => 1, COMPRESSION => 'SNAPPY', BLOOMFILTER => 'ROW'},
  {NAME => 'geo',   VERSIONS => 1, COMPRESSION => 'SNAPPY'},
  {NAME => 'cart',  VERSIONS => 1, COMPRESSION => 'SNAPPY'}

# TABLE 2: Product performance (date-bucketed metrics)
create 'product_performance',
  {NAME => 'views', VERSIONS => 1, COMPRESSION => 'SNAPPY'},
  {NAME => 'cart',  VERSIONS => 1, COMPRESSION => 'SNAPPY'},
  {NAME => 'sales', VERSIONS => 1, COMPRESSION => 'SNAPPY'}

# TABLE 3: Raw page events (append-only clickstream)
create 'page_events',
  {NAME => 'ev', VERSIONS => 1, COMPRESSION => 'SNAPPY', BLOOMFILTER => 'ROWCOL'}

# ── Sample HBase Shell Queries ────────────────────────────────────────────────

# Query 1: Retrieve last 5 sessions for a specific user
# (reverse timestamp key means first 5 rows = most recent)
scan 'user_sessions', {
  STARTROW => 'user_000042#',
  STOPROW  => 'user_000042#~',
  LIMIT    => 5
}

# Query 2: Get product performance for prod_00123 over last 30 days
scan 'product_performance', {
  STARTROW => 'prod_00123#20250215',
  STOPROW  => 'prod_00123#20250316'
}

# Query 3: Count rows in user_sessions table
count 'user_sessions', {INTERVAL => 100000}

# Query 4: Get a specific session's metadata
get 'user_sessions', 'user_000042#<reverse_ts>', {COLUMN => 'meta'}

# Query 5: Filter sessions by conversion status (server-side filter)
import org.apache.hadoop.hbase.filter.SingleColumnValueFilter
import org.apache.hadoop.hbase.filter.CompareFilter
import org.apache.hadoop.hbase.util.Bytes

scan 'user_sessions', {
  FILTER => "SingleColumnValueFilter('meta', 'conversion_status',
             =, 'binary:converted')",
  LIMIT  => 100
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATED HBASE STORE  (dict-based, mirrors HBase row/column structure)
# ─────────────────────────────────────────────────────────────────────────────

class SimulatedHBase:
    """
    Dict-based simulation of HBase's row/column-family/qualifier model.
    Produces identical results to real HBase client code.
    Used when a live HBase cluster is unavailable.
    """

    def __init__(self):
        self.tables = defaultdict(dict)

    def put(self, table: str, row_key: str, data: dict):
        """data = {'family:qualifier': value}"""
        if row_key not in self.tables[table]:
            self.tables[table][row_key] = {}
        self.tables[table][row_key].update(data)

    def get(self, table: str, row_key: str, column_family: str = None):
        row = self.tables[table].get(row_key, {})
        if column_family:
            return {k: v for k, v in row.items() if k.startswith(f"{column_family}:")}
        return row

    def scan(self, table: str, start_row: str = None, stop_row: str = None, limit: int = None):
        rows = sorted(self.tables[table].items())
        result = []
        for rk, cols in rows:
            if start_row and rk < start_row:
                continue
            if stop_row and rk >= stop_row:
                break
            result.append((rk, cols))
            if limit and len(result) >= limit:
                break
        return result

    def count(self, table: str):
        return len(self.tables[table])


def epoch_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def reverse_ts(dt: datetime) -> str:
    """HBase reverse-timestamp: Long.MAX_VALUE - epochMs, zero-padded."""
    MAX_LONG = 9_223_372_036_854_775_807
    return str(MAX_LONG - epoch_ms(dt)).zfill(19)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_sessions_to_hbase(hbase: SimulatedHBase, max_sessions: int = 5000):
    """
    Load session data into the user_sessions table.
    Row key: user_id#reverse_timestamp  (ensures most-recent-first scans)
    """
    print(f"\n[HBase] Loading up to {max_sessions:,} sessions into user_sessions...")
    loaded = 0

    import glob
    session_files = sorted(glob.glob(os.path.join(DATA_DIR, "sessions_*.json")))

    for sf in session_files:
        with open(sf) as f:
            sessions = json.load(f)

        for session in sessions:
            if loaded >= max_sessions:
                break

            start_dt = datetime.fromisoformat(session["start_time"])
            rev_ts   = reverse_ts(start_dt)
            row_key  = f"{session['user_id']}#{rev_ts}"

            device   = session.get("device_profile", {})
            geo      = session.get("geo_data", {})
            cart     = session.get("cart_contents", {})
            cart_ids = list(cart.keys())
            cart_val = sum(
                v.get("quantity", 0) * v.get("price", 0)
                for v in cart.values()
            )

            hbase.put("user_sessions", row_key, {
                # meta column family
                "meta:start_time":         session.get("start_time", ""),
                "meta:end_time":           session.get("end_time", ""),
                "meta:duration_seconds":   str(session.get("duration_seconds", 0)),
                "meta:device_type":        device.get("type", ""),
                "meta:device_os":          device.get("os", ""),
                "meta:browser":            device.get("browser", ""),
                "meta:referrer":           session.get("referrer", ""),
                "meta:conversion_status":  session.get("conversion_status", ""),
                # geo column family
                "geo:city":                geo.get("city", ""),
                "geo:state":               geo.get("state", ""),
                "geo:country":             geo.get("country", ""),
                "geo:ip_address":          geo.get("ip_address", ""),
                # cart column family
                "cart:product_ids":        json.dumps(cart_ids),
                "cart:total_value":        str(round(cart_val, 2))
            })
            loaded += 1

        if loaded >= max_sessions:
            break

    print(f"  ✓ Loaded {hbase.count('user_sessions'):,} rows into user_sessions")


def build_product_performance(hbase: SimulatedHBase, max_sessions: int = 5000):
    """
    Aggregate product view metrics from session page_views into
    product_performance table (row key: product_id#YYYYMMDD).
    """
    print("[HBase] Building product_performance metrics...")
    # In-memory aggregation before writing to simulated HBase
    daily_views = defaultdict(lambda: {"count": 0, "unique_users": set(), "dur": 0})

    import glob
    session_files = sorted(glob.glob(os.path.join(DATA_DIR, "sessions_*.json")))
    processed = 0

    for sf in session_files:
        with open(sf) as f:
            sessions = json.load(f)
        for session in sessions:
            if processed >= max_sessions:
                break
            user_id = session["user_id"]
            for pv in session.get("page_views", []):
                if pv.get("page_type") == "product_detail" and pv.get("product_id"):
                    prod_id  = pv["product_id"]
                    try:
                        dt   = datetime.fromisoformat(pv["timestamp"])
                        date = dt.strftime("%Y%m%d")
                    except Exception:
                        continue
                    key = f"{prod_id}#{date}"
                    daily_views[key]["count"]          += 1
                    daily_views[key]["unique_users"].add(user_id)
                    daily_views[key]["dur"]            += pv.get("view_duration", 0)
            processed += 1
        if processed >= max_sessions:
            break

    for key, agg in daily_views.items():
        n = agg["count"]
        avg_dur = round(agg["dur"] / n, 1) if n else 0
        hbase.put("product_performance", key, {
            "views:count":        str(n),
            "views:unique_users": str(len(agg["unique_users"])),
            "views:avg_dur_sec":  str(avg_dur)
        })

    print(f"  ✓ Loaded {hbase.count('product_performance'):,} product-day rows")


# ─────────────────────────────────────────────────────────────────────────────
# QUERIES
# ─────────────────────────────────────────────────────────────────────────────

def query_user_sessions(hbase: SimulatedHBase, user_id: str, n: int = 5):
    """
    Retrieve the N most-recent sessions for a user.
    Equivalent HBase shell command:
        scan 'user_sessions', {STARTROW=>'user_X#', STOPROW=>'user_X#~', LIMIT=>N}
    """
    print(f"\n── Query: Last {n} sessions for {user_id} ──────────────────────")
    rows = hbase.scan(
        "user_sessions",
        start_row=f"{user_id}#",
        stop_row=f"{user_id}#~",   # '~' is lexicographically greater than any digit
        limit=n
    )
    if not rows:
        print(f"  No sessions found for {user_id}")
        return []
    for rk, cols in rows:
        start  = cols.get("meta:start_time", "")[:19]
        status = cols.get("meta:conversion_status", "")
        device = cols.get("meta:device_type", "")
        dur    = cols.get("meta:duration_seconds", "0")
        ref    = cols.get("meta:referrer", "")
        print(f"  {start}  [{status:<10}]  device={device:<8}  dur={dur}s  via={ref}")
    return rows


def query_product_performance(hbase: SimulatedHBase, product_id: str,
                               start_date: str = "20250101", end_date: str = "20251231"):
    """
    Retrieve daily view metrics for a product across a date range.
    Equivalent HBase scan:
        scan 'product_performance', {STARTROW=>'{prod}#{start}', STOPROW=>'{prod}#{end}'}
    """
    print(f"\n── Query: Performance for {product_id} ({start_date}–{end_date}) ─")
    rows = hbase.scan(
        "product_performance",
        start_row=f"{product_id}#{start_date}",
        stop_row=f"{product_id}#{end_date}"
    )
    if not rows:
        print(f"  No performance data for {product_id}")
        return []
    total_views = 0
    for rk, cols in rows:
        date   = rk.split("#")[1]
        count  = int(cols.get("views:count", 0))
        unique = cols.get("views:unique_users", "0")
        dur    = cols.get("views:avg_dur_sec", "0")
        total_views += count
        print(f"  {date}  views={count:4d}  unique_users={unique:4s}  avg_dur={dur}s")
    print(f"  Total views in range: {total_views}")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# REAL HAPPYBASE CLIENT CODE (when HBase is available)
# ─────────────────────────────────────────────────────────────────────────────

HAPPYBASE_CLIENT_CODE = '''
# Real HBase client using happybase (pip install happybase)
# Requires HBase Thrift server running on port 9090
# docker run -d -p 9090:9090 -p 16000:16000 harisekhon/hbase

import happybase

connection = happybase.Connection("localhost", port=9090)
connection.open()

# Create tables (run once)
connection.create_table("user_sessions", {
    "meta": {"max_versions": 1, "compression": "SNAPPY"},
    "geo":  {"max_versions": 1},
    "cart": {"max_versions": 1}
})

# Write a session row
table = connection.table("user_sessions")
table.put(
    b"user_000042#<reverse_ts>",
    {
        b"meta:conversion_status": b"converted",
        b"meta:device_type":       b"mobile",
        b"meta:duration_seconds":  b"919",
        b"geo:country":            b"US",
        b"cart:total_value":       b"259.98"
    }
)

# Scan recent sessions for a user (most-recent-first due to reverse key)
for key, data in table.scan(
    row_start=b"user_000042#",
    row_stop=b"user_000042#~",
    limit=5
):
    print(key.decode(), {k.decode(): v.decode() for k, v in data.items()})

connection.close()
'''


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("HBase Schema & Query Demonstration")
    print("(Simulation mode — mirrors real HBase API behaviour)")
    print("=" * 60)

    hbase = SimulatedHBase()

    # Load data
    load_sessions_to_hbase(hbase, max_sessions=10000)
    build_product_performance(hbase, max_sessions=10000)

    # Demonstrate queries
    # Pick a real user_id from the data
    with open(os.path.join(DATA_DIR, "users.json")) as f:
        users = json.load(f)
    sample_user    = users[42]["user_id"]
    sample_product = "prod_00050"

    query_user_sessions(hbase, sample_user, n=5)
    query_product_performance(hbase, sample_product)

    print("\n── Schema DDL (HBase Shell Commands) ────────────────────────")
    print(HBASE_SHELL_COMMANDS)
    print("\n── Schema Design Rationale ──────────────────────────────────")
    print(HBASE_SCHEMA_RATIONALE)
    print("\n── Real HBase Client Code (happybase) ────────────────────────")
    print(HAPPYBASE_CLIENT_CODE)

    # Save metadata for report
    out = os.path.join(DATA_DIR, "hbase_metadata.json")
    with open(out, "w") as f:
        json.dump({
            "user_sessions_rows":      hbase.count("user_sessions"),
            "product_performance_rows": hbase.count("product_performance"),
            "sample_user":             sample_user,
            "sample_product":          sample_product
        }, f, indent=2)
    print(f"\n  ✓ Metadata saved to {out}")


if __name__ == "__main__":
    main()
