import os, psycopg
from dotenv import load_dotenv
load_dotenv()
with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as c:
    c.execute("""CREATE TABLE IF NOT EXISTS conversations (
        conv_id TEXT PRIMARY KEY,
        updated_at TIMESTAMPTZ DEFAULT now(),
        data JSONB NOT NULL)""")
    print("table created OK")
    print(c.execute("SELECT tablename FROM pg_tables WHERE schemaname='public'").fetchall())