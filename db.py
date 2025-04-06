from psycopg_pool import AsyncConnectionPool
import os

DB_URI = os.getenv("DB_URI")

pool = AsyncConnectionPool(
    conninfo=DB_URI,
    max_size=5,
    kwargs={"autocommit": True, "prepare_threshold": 0},
)
