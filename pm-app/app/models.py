from datetime import datetime, timezone
from sqlalchemy import Column, BigInteger, Integer, String, Float, Text, DateTime, Index, func
from sqlalchemy.dialects.postgresql import JSONB, BIGINT
from sqlalchemy.types import JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

# Use JSONB for PostgreSQL, JSON for SQLite
JSONType = JSON().with_variant(JSONB(), 'postgresql')

# Use INTEGER for SQLite (autoincrement), BIGINT for PostgreSQL
BigIntegerType = Integer().with_variant(BIGINT(), 'postgresql')


class Telemetry(Base):
    """
    CREATE TABLE telemetry (
        product_id TEXT,
        unit_id TEXT,
        timestamp TEXT,
        step_index INTEGER,
        engine_type TEXT,
        air_temperature_K DOUBLE PRECISION,
        process_temperature_K DOUBLE PRECISION,
        rotational_speed_rpm DOUBLE PRECISION,
        torque_Nm DOUBLE PRECISION,
        tool_wear_min DOUBLE PRECISION,
        is_failure INTEGER,
        failure_type TEXT,
        synthetic_RUL DOUBLE PRECISION,
        PRIMARY KEY (product_id, unit_id, timestamp)
    );
    """
    __tablename__ = 'telemetry'
    __table_args__ = (
        Index('telemetry_unit_idx', 'unit_id'),
        Index('telemetry_failure_idx', 'is_failure'),
    )

    product_id = Column(Text, primary_key=True)
    unit_id = Column(Text, primary_key=True)
    timestamp = Column(Text, primary_key=True)
    step_index = Column(Integer)
    engine_type = Column(Text)
    air_temperature_K = Column(Float)
    process_temperature_K = Column(Float)
    rotational_speed_rpm = Column(Float)
    torque_Nm = Column(Float)
    tool_wear_min = Column(Float)
    is_failure = Column(Integer)
    failure_type = Column(Text)
    synthetic_RUL = Column(Float)


class RetrainPointer(Base):
    """
    CREATE TABLE retrain_pointer (
        model_name TEXT PRIMARY KEY,
        last_retrain_ts TIMESTAMPTZ DEFAULT '1970-01-01T00:00:00Z',
        last_retrain_id BIGINT DEFAULT 0
    );
    """
    __tablename__ = 'retrain_pointer'

    model_name = Column(Text, primary_key=True)
    last_retrain_ts = Column(DateTime(timezone=True), server_default='1970-01-01 00:00:00+00')
    last_retrain_id = Column(BigInteger, default=0)


class ModelArtifact(Base):
    """
    CREATE TABLE model_artifact (
        id BIGSERIAL PRIMARY KEY,
        model_name TEXT NOT NULL,
        version TEXT NOT NULL,
        metadata JSONB,
        promoted_at TIMESTAMPTZ DEFAULT NOW()
    );
    """
    __tablename__ = 'model_artifact'
    __table_args__ = (
        Index('model_artifact_idx', 'model_name', 'promoted_at'),
    )

    id = Column(BigIntegerType, primary_key=True, autoincrement=True)
    model_name = Column(Text, nullable=False)
    version = Column(Text, nullable=False)
    model_metadata = Column('metadata', JSONType)
    promoted_at = Column(DateTime(timezone=True), server_default=func.now())


class MaintenanceSchedule(Base):
    """
    CREATE TABLE maintenance_schedule (
        id BIGSERIAL PRIMARY KEY,
        schedule_id TEXT NOT NULL,
        product_id TEXT NOT NULL,
        unit_id TEXT NOT NULL,
        recommended_start TIMESTAMPTZ NOT NULL,
        recommended_end TIMESTAMPTZ NOT NULL,
        reason TEXT NOT NULL,
        risk_score DOUBLE PRECISION,
        model_version TEXT,
        actions JSONB,
        constraints_applied JSONB,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        status TEXT DEFAULT 'PENDING'
    );
    """
    __tablename__ = 'maintenance_schedule'
    __table_args__ = (
        Index('maintenance_schedule_unit_idx', 'product_id', 'unit_id', 'recommended_start'),
    )

    id = Column(BigIntegerType, primary_key=True, autoincrement=True)
    schedule_id = Column(Text, nullable=False)
    product_id = Column(Text, nullable=False)
    unit_id = Column(Text, nullable=False)
    recommended_start = Column(DateTime(timezone=True), nullable=False)
    recommended_end = Column(DateTime(timezone=True), nullable=False)
    reason = Column(Text, nullable=False)
    risk_score = Column(Float)
    model_version = Column(Text)
    actions = Column(JSONType)
    constraints_applied = Column(JSONType)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    status = Column(Text, default='PENDING')


class ConversationHistory(Base):
    """
    LangChain conversation memory storage.
    
    This table is used by LangChain's PostgresChatMessageHistory.
    Schema follows LangChain's expected format with 'message' column.
    """
    __tablename__ = 'conversation_history'
    __table_args__ = (
        Index('ix_conversation_history_session_id', 'session_id'),
    )
    
    id = Column(BigIntegerType, primary_key=True, autoincrement=True)
    session_id = Column(Text, nullable=False)
    message = Column(Text, nullable=False)  # LangChain stores complete message as JSONB