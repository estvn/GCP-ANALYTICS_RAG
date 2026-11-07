import uuid

import psycopg
import pytest

from rag.store import init_schema

ADMIN = "postgresql://rag:rag@127.0.0.1:5433/postgres"


@pytest.fixture
def conn():
    """A throwaway database on the local pgvector container, dropped after the test."""
    name = f"rag_test_{uuid.uuid4().hex[:8]}"
    try:
        admin = psycopg.connect(ADMIN, autocommit=True)
    except psycopg.OperationalError:
        pytest.skip("postgres not running")
    admin.execute(f'CREATE DATABASE "{name}"')
    c = psycopg.connect(ADMIN.rsplit("/", 1)[0] + "/" + name)
    init_schema(c)
    yield c
    c.close()
    admin.execute(f'DROP DATABASE "{name}" WITH (FORCE)')
    admin.close()
