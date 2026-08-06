from sqlalchemy import create_engine, URL, text
import pandas as pd

DB_URL = URL.create(
    "mysql+pymysql",           # dialect+driver, instead of "postgresql+pg8000"
    username="root",
    password="985196",         # plain text — no URL-encoding needed
    host="localhost",
    port=3306,
    database="olist",
)

engine = create_engine(DB_URL)

q = '''
    select * from orders
    limit 5
'''
#SQL 
# with engine.connect() as conn:
#     result = conn.execute(text(q))
#     rows = result.fetchall()

#     for row in rows:
#         print(row)

#PYTHON
res = pd.read_sql(q, engine)

print(res)