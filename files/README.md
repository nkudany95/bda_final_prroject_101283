# AUCA Big Data Analytics Final Project
**Distributed Multi-Model Analytics for E-Commerce Data**

Student: UMWALI MWIZA Rita Lys Cleria | ID: 20251MBI054

## Project Structure
```
ecommerce_analytics/
├── data/                         # Dataset (generated, not committed)
│   ├── dataset_generator_scaled.py  # Scaled generator (50K sessions)
│   ├── users.json                   # 1,000 user profiles
│   ├── products.json                # 500 products
│   ├── categories.json              # 25 categories
│   ├── transactions.json            # 10,000 transactions
│   └── sessions_*.json              # 50,000 sessions (5 chunks)
├── mongodb/
│   └── mongodb_analytics.py         # Schema design + 2 aggregation pipelines
├── hbase/
│   └── hbase_schema.py              # DDL, row-key design, query simulation
├── spark/
│   └── spark_analytics.py           # Cleaning, co-purchase, cohort, Spark SQL
├── integration/
│   └── integration_analytics.py     # CLV cross-store integration
├── visualizations/
│   ├── visualizations.py            # 5 charts (matplotlib/seaborn)
│   └── chart*.png                   # Generated charts
└── report/
    └── AUCA_BigData_Final_Report.docx
```

## Quick Start

### 1. Install dependencies
```bash
pip install faker numpy pandas matplotlib seaborn plotly pymongo pyspark
```

### 2. Generate dataset
```bash
cd data/
python dataset_generator_scaled.py        # ~100MB, runs in ~3 min
# For full scale (requires 16GB+ RAM):
# python dataset_generator.py             # ~3-5GB, runs in ~30-60 min
```

### 3. Run MongoDB analytics (requires MongoDB running on localhost:27017)
```bash
cd mongodb/
python mongodb_analytics.py
```

### 4. Run HBase schema + queries (simulation mode, no HBase needed)
```bash
cd hbase/
python hbase_schema.py
# For real HBase: docker run -d -p 9090:9090 -p 16000:16000 harisekhon/hbase
```

### 5. Run Spark analytics (requires pyspark)
```bash
cd spark/
python spark_analytics.py
# Or: spark-submit spark_analytics.py
```

### 6. Run integration analytics
```bash
cd integration/
python integration_analytics.py
```

### 7. Generate visualizations
```bash
cd visualizations/
python visualizations.py
```

## Key Results
- Total revenue (90-day window, scaled dataset): **$5,664,676**
- Email channel conversion rate: **21.1%** (highest across all channels)
- High-CLV customers show **23% session conversion rate** vs 17.7% for Low-CLV
- Top device-OS combination: **Mobile-iOS** by session volume; **Desktop-Windows** by order value

## Technology Stack
| Component | Technology | Version |
|-----------|-----------|---------|
| Document store | MongoDB | 7.x |
| Wide-column store | HBase | 2.x |
| Distributed processing | Apache Spark / PySpark | 3.5+ |
| Data generation | Python / Faker | 3.10+ |
| Visualization | Matplotlib / Seaborn | Latest |
