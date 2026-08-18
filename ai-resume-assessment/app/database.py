import logging
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from app.config import settings

logger = logging.getLogger("ai_resume_assessment.database")

from sqlalchemy.pool import NullPool

# Normalize Postgres driver for async SQLAlchemy if postgresql:// is passed
db_url = settings.DATABASE_URL
if db_url.startswith("postgresql://"):
    db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

engine = create_async_engine(db_url, echo=False, future=True, poolclass=NullPool)

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

Base = declarative_base()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency injection for obtaining an async database session.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"Database session exception: {str(e)}")
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db(drop_existing: bool = False):
    """
    Creates database tables asynchronously if they do not exist.
    Optionally drops existing tables to apply schema modifications.
    """
    logger.info(f"Initializing database tables (drop_existing={drop_existing})...")
    async with engine.begin() as conn:
        if drop_existing:
            await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()
    logger.info("Database tables initialization completed.")

