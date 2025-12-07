import os
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import quote_plus


def _build_database_url() -> str:
    """
    Build database URL from separate environment variables for Cloud SQL compatibility.
    
    Supports two modes:
    1. Direct DATABASE_URL (traditional)
    2. Separate params: DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME, DB_CONNECTION_NAME
    
    For Cloud SQL with Unix socket (Cloud Run):
        DB_CONNECTION_NAME=project:region:instance
        DB_SOCKET_PATH=/cloudsql (optional, defaults to /cloudsql)
    
    For Cloud SQL with TCP (local dev with Cloud SQL Proxy):
        DB_HOST=127.0.0.1
        DB_PORT=5432
    """
    # Check for direct DATABASE_URL first
    if os.getenv('DATABASE_URL'):
        return os.getenv('DATABASE_URL')
    
    # Build from separate params
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    db_name = os.getenv('DB_NAME', 'pm_database')
    db_host = os.getenv('DB_HOST')
    db_port = os.getenv('DB_PORT', '5432')
    db_connection_name = os.getenv('DB_CONNECTION_NAME')  # Cloud SQL instance connection name
    db_socket_path = os.getenv('DB_SOCKET_PATH', '/cloudsql')  # Unix socket path for Cloud Run
    
    # If no credentials provided, fall back to SQLite
    if not db_user:
        return 'sqlite:///pm_app.db'
    
    # URL-encode password to handle special characters
    encoded_password = quote_plus(db_password) if db_password else ''
    
    # Cloud SQL Unix socket connection (for Cloud Run)
    if db_connection_name:
        # Format: postgresql+pg8000://user:pass@/dbname?unix_sock=/cloudsql/project:region:instance/.s.PGSQL.5432
        socket_path = f"{db_socket_path}/{db_connection_name}/.s.PGSQL.5432"
        return f"postgresql+pg8000://{db_user}:{encoded_password}@/{db_name}?unix_sock={quote_plus(socket_path)}"
    
    # TCP connection (local dev with Cloud SQL Proxy or direct connection)
    if db_host:
        return f"postgresql+pg8000://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"
    
    # Fallback to SQLite
    return 'sqlite:///pm_app.db'


@dataclass
class DatabaseConfig:
    # Connection parameters (separate for Cloud SQL / Cloud Run)
    user: str | None = field(default_factory=lambda: os.getenv('DB_USER'))
    password: str | None = field(default_factory=lambda: os.getenv('DB_PASSWORD'))
    host: str | None = field(default_factory=lambda: os.getenv('DB_HOST'))
    port: str = field(default_factory=lambda: os.getenv('DB_PORT', '5432'))
    name: str = field(default_factory=lambda: os.getenv('DB_NAME', 'pm_database'))
    connection_name: str | None = field(default_factory=lambda: os.getenv('DB_CONNECTION_NAME'))  # Cloud SQL instance
    socket_path: str = field(default_factory=lambda: os.getenv('DB_SOCKET_PATH', '/cloudsql'))
    
    # Built URL (computed from above or DATABASE_URL env var)
    url: str = field(default_factory=_build_database_url)
    
    # Pool settings
    pool_size: int = field(default_factory=lambda: int(os.getenv('DB_POOL_SIZE', '10')))
    max_overflow: int = field(default_factory=lambda: int(os.getenv('DB_MAX_OVERFLOW', '20')))
    echo: bool = field(default_factory=lambda: os.getenv('DB_ECHO', 'false').lower() == 'true')


@dataclass
class ModelStorageConfig:
    backend: Literal['local', 'gcs'] = field(default_factory=lambda: os.getenv('MODEL_STORAGE_BACKEND', 'local'))
    local_path: str = field(default_factory=lambda: os.getenv('MODEL_STORAGE_PATH', './artifacts'))
    gcs_bucket: str | None = field(default_factory=lambda: os.getenv('GCS_MODEL_BUCKET'))


@dataclass
class GeminiConfig:
    api_key: str | None = field(default_factory=lambda: os.getenv('GEMINI_API_KEY'))
    model_name: str = field(default_factory=lambda: os.getenv('GEMINI_MODEL_NAME', 'gemini-2.5-pro-exp'))
    api_url: str = field(default_factory=lambda: os.getenv('GEMINI_API_URL', 'https://generativelanguage.googleapis.com/v1beta'))
    timeout: int = field(default_factory=lambda: int(os.getenv('GEMINI_TIMEOUT', '60')))
    stt_model: str = field(default_factory=lambda: os.getenv('GEMINI_STT_MODEL', 'gemini-2.5-pro'))
    max_audio_bytes: int = field(default_factory=lambda: int(os.getenv('MAX_AUDIO_MB', '10')) * 1024 * 1024)


@dataclass
class AppConfig:
    flask_env: str = field(default_factory=lambda: os.getenv('FLASK_ENV', 'development'))
    secret_key: str = field(default_factory=lambda: os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production'))
    log_level: str = field(default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO'))
    log_format: Literal['text', 'json'] = field(default_factory=lambda: os.getenv('LOG_FORMAT', 'text'))
    cors_origins: list[str] = field(default_factory=lambda: os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://localhost:5173').split(','))
    
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    model_storage: ModelStorageConfig = field(default_factory=ModelStorageConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    
    @property
    def is_production(self) -> bool:
        return self.flask_env == 'production'
    
    @property
    def is_development(self) -> bool:
        return self.flask_env == 'development'


def load_config() -> AppConfig:
    return AppConfig()
