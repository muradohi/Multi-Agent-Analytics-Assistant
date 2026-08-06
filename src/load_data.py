
import sys
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text, URL
import os

# ======================================================================
# CONFIG
# ======================================================================
USE_MYSQL = False   # True -> your local MySQL;  False -> local SQLite file

DB_URL = URL.create(
    "mysql+pymysql",
    username="root",
    password= os.getenv("DB_PASSWORD"),
    host = "localhost",
    port = 3306,
    database= "olist"
)

if USE_MYSQL:
    DB_URL = URL.create(
    "mysql+pymysql",
    username="root",
    password= os.getenv("DB_PASSWORD"),
    host = "localhost",
    port = 3306,
    database= "olist"
)
else:
    DB_URL = "sqlite:///olist.db"

CURR_DIR = Path(__file__).parent.parent
DATA_DIR = Path("data")

DATA_FILE = os.path.join(CURR_DIR, DATA_DIR)

# d = os.path.join(DATA_FILE, filename)
# data = pd.read_csv(d)
# print(data.head(5))

# print(CURR_DIR)
# print(DATA_DIR)
# print(DATA_FILE)

# Map: the table name your agent will query  ->  the Olist CSV filename.
# These 5 give you real joins (orders->order_items->products) plus review TEXT.
# Add sellers/payments/geolocation later only if a question needs them.
FILES = {
    "orders":        "olist_orders_dataset.csv",
    "order_items":   "olist_order_items_dataset.csv",
    "products":      "olist_products_dataset.csv",
    "customers":     "olist_customers_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
}


def main():
    engine = create_engine(DB_URL)

    # ---- load each CSV into its own table ----
    for table_name, filename in FILES.items():
        path = os.path.join(DATA_FILE, filename)
        # if not path.exists():
        #     print(f"ERROR: {path} not found. Check DATA_DIR and the filename.")
        #     sys.exit(1)

        df = pd.read_csv(path)
        # normalize column names so the agent reads clean, consistent names
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        df = df.drop_duplicates()
        df.to_sql(table_name, engine, if_exists="replace", index=False)
        print(f"loaded {table_name:15} {len(df):>7} rows")

    # ---- VERIFY with a real 2-table join before trusting anything ----
    verify_query = """
        SELECT p.product_category_name,
               COUNT(*)        AS num_items,
               AVG(oi.price)   AS avg_price
        FROM order_items oi
        JOIN products p ON oi.product_id = p.product_id
        GROUP BY p.product_category_name
        ORDER BY num_items DESC
        LIMIT 5
    """
    print("\n=== VERIFICATION (top 5 categories by items sold) ===")
    with engine.connect() as conn:
        for row in conn.execute(text(verify_query)).fetchall():
            print("  ", tuple(row))

    print("\nDone. Tables are loaded and joins work. Ready for the SQL agent.")


if __name__ == "__main__":
    main()