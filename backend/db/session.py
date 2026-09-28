import os
from collections.abc import Generator
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


def get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not configured. Set it to a PostgreSQL connection string before using database features."
        )
    return database_url


def create_db_engine(database_url: str | None = None, *, echo: bool = False) -> Engine:
    resolved_url = database_url or get_database_url()
    return create_engine(resolved_url, echo=echo, future=True, pool_pre_ping=True)


def _build_session_factory(database_url: str | None = None):
    return sessionmaker(
        bind=create_db_engine(database_url),
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )


@lru_cache(maxsize=1)
def _default_session_factory():
    return _build_session_factory()


def get_session_factory(database_url: str | None = None):
    if database_url is None:
        return _default_session_factory()
    return _build_session_factory(database_url)


def get_db_session() -> Generator[Session, None, None]:
    session_factory = get_session_factory()
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
