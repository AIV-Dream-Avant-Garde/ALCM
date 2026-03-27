"""Async SQLAlchemy engine, session, and base model."""
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from .config import get_settings

settings = get_settings()

engine = create_async_engine(settings.database_url, echo=False)
async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db():
    """Dependency yielding an async DB session."""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db():
    """Verify database connectivity on startup.

    Schema creation is handled by Alembic migrations, not create_all.
    This function only verifies the connection works.
    """
    async with engine.begin() as conn:
        await conn.execute(text("SELECT 1"))
