from contextlib import contextmanager
from typing import Generator
import os
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from app.config import load_config


_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None


def _get_cloud_sql_connector():
    """
    Create a Cloud SQL Python Connector for secure connections.
    Returns None if not using Cloud SQL connection name.
    """
    connection_name = os.getenv('DB_CONNECTION_NAME')
    if not connection_name:
        return None
    
    try:
        from google.cloud.sql.connector import Connector
        return Connector()
    except ImportError:
        return None


def _create_cloud_sql_engine(config) -> Engine | None:
    """
    Create engine using Cloud SQL Python Connector (recommended for Cloud Run).
    Returns None if Cloud SQL connection is not configured.
    """
    connection_name = os.getenv('DB_CONNECTION_NAME')
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    db_name = os.getenv('DB_NAME', 'pm_database')
    
    if not all([connection_name, db_user, db_password]):
        return None
    
    try:
        from google.cloud.sql.connector import Connector
        import pg8000
        
        connector = Connector()
        
        def getconn():
            return connector.connect(
                connection_name,
                "pg8000",
                user=db_user,
                password=db_password,
                db=db_name,
            )
        
        engine = create_engine(
            "postgresql+pg8000://",
            creator=getconn,
            echo=config.database.echo,
            pool_size=config.database.pool_size,
            max_overflow=config.database.max_overflow,
            pool_pre_ping=True,
        )
        
        return engine
    except ImportError:
        # Fall back to URL-based connection if connector not available
        return None
    except Exception:
        return None


def init_db(database_url: str | None = None, **engine_kwargs) -> None:
    global _engine, _SessionLocal
    
    config = load_config()
    
    # Try Cloud SQL Python Connector first (best for Cloud Run)
    if not database_url and os.getenv('DB_CONNECTION_NAME'):
        cloud_engine = _create_cloud_sql_engine(config)
        if cloud_engine:
            _engine = cloud_engine
            _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
            return
    
    # Fall back to URL-based connection
    db_url = database_url or config.database.url
    
    engine_options = {
        'echo': config.database.echo,
    }
    
    if db_url.startswith('sqlite'):
        engine_options.update({
            'connect_args': {'check_same_thread': False},
            'poolclass': StaticPool,
        })
        
        @event.listens_for(Engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    else:
        engine_options.update({
            'pool_size': config.database.pool_size,
            'max_overflow': config.database.max_overflow,
            'pool_pre_ping': True,
        })
    
    engine_options.update(engine_kwargs)
    
    _engine = create_engine(db_url, **engine_options)
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)


def get_engine() -> Engine:
    if _engine is None:
        init_db()
    return _engine


def SessionLocal() -> Session:
    if _SessionLocal is None:
        init_db()
    return _SessionLocal()


@contextmanager
def get_db() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_db(new_engine: Engine | None = None) -> None:
    global _engine, _SessionLocal
    
    if new_engine:
        _engine = new_engine
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    else:
        _engine = None
        _SessionLocal = None
