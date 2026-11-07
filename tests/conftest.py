import uuid

import psycopg
import pytest

from rag.store import init_schema

ADMIN = "postgresql://rag:rag@127.0.0.1:5433/postgres"


@pytest.fixture
def make_db():
    """Factory of throwaway databases on the local pgvector container; all are dropped after the test."""
    try:
        admin = psycopg.connect(ADMIN, autocommit=True)
    except psycopg.OperationalError:
        pytest.skip("postgres not running")
    created = []

    def make(dim: int = 384):
        name = f"rag_test_{uuid.uuid4().hex[:8]}"
        admin.execute(f'CREATE DATABASE "{name}"')
        c = psycopg.connect(ADMIN.rsplit("/", 1)[0] + "/" + name)
        init_schema(c, dim=dim)
        created.append((name, c))
        return c

    yield make
    for name, c in created:
        c.close()
        admin.execute(f'DROP DATABASE "{name}" WITH (FORCE)')
    admin.close()


@pytest.fixture
def conn(make_db):
    return make_db()
