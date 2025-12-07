# Predictive Maintenance System

End-to-end machine learning pipeline and inference backend for equipment failure prediction, remaining useful life (RUL) forecasting, and maintenance schedule optimization.

## Project Overview

This system provides:

- **Data preprocessing**: Dataset reconstruction, stratified sampling, time-series generation
- **Model training**: XGBoost classification and regression with hyperparameter optimization
- **Inference backend**: Flask REST API with PostgreSQL/SQLite database support
- **AI copilot**: Gemini 2.5 Pro integration for natural language maintenance queries
- **Scheduling**: Constraint-based maintenance optimization with multi-unit support

## Repository Structure

```
machine-learning/
├── data/                              # Raw datasets
│   ├── predictive_maintenance.csv     # Original static dataset
│   ├── marine_engine_data.csv
│   └── equipment_anomaly_data.csv
├── preprocessed/                      # Processed datasets
│   ├── predictive_maintenance_subset_400_machines.csv
│   └── predictive_maintenance_timeseries.csv
├── artifacts/                         # Trained model files
│   ├── xgb_classifier.joblib          # Failure classification (XGBoost)
│   ├── xgb_regressor.joblib           # RUL prediction (XGBoost)
│   ├── lstm_classifier.keras          # Alternative LSTM classifier
│   ├── lstm_regressor.keras           # Alternative LSTM regressor
│   ├── scaler.joblib                  # Feature standardization
│   ├── encoder_engine_type.joblib     # Categorical encoder
│   └── *.csv                          # Model metadata and metrics
├── scripts/                           # Preprocessing utilities
│   ├── dataset_reconstruction.py      # Stratified sampling and subset creation
│   └── generate_timeseries.py         # Time-series sequence generation
├── sql/                               # Database schema
│   └── schema.sql                     # PostgreSQL/SQLite schema definition
├── pm-app/                            # Flask inference backend
│   ├── app/
│   │   ├── __init__.py                # Application factory
│   │   ├── config.py                  # Configuration loader
│   │   ├── db.py                      # Database session management
│   │   ├── models.py                  # SQLAlchemy ORM models
│   │   ├── routes/                    # API endpoints
│   │   │   ├── classification.py      # POST /predict/classification
│   │   │   ├── forecast.py            # POST /predict/forecast
│   │   │   ├── optimizer.py           # POST /optimizer/schedule
│   │   │   ├── models_mgmt.py         # GET /models, POST /retrain/run
│   │   │   ├── maintenance.py         # GET /maintenance/schedule
│   │   │   ├── health.py              # GET /health
│   │   │   └── copilot_tools.py       # POST /copilot/chat
│   │   └── services/                  # Business logic
│   │       ├── models_loader.py       # Model caching and loading
│   │       ├── classification_service.py
│   │       ├── forecast_service.py
│   │       ├── optimizer_service.py
│   │       ├── retrain_service.py     # Incremental retraining
│   │       ├── copilot_service.py     # Gemini direct API client
│   │       ├── langchain_agent_service.py  # LangChain agent orchestration
│   │       └── agent_tools.py         # LangChain tool definitions
│   ├── scripts/                       # Database utilities
│   │   ├── init_db.py                 # Initialize database schema
│   │   └── migrate_to_legacy_schema.py # Schema migration tool
│   ├── tests/                         # Pytest test suite
│   │   ├── conftest.py                # Test fixtures and setup
│   │   ├── test_classification_api.py
│   │   ├── test_forecast_api.py
│   │   └── test_optimizer_api.py
│   ├── Dockerfile                     # Multi-stage container build
│   ├── docker-compose.yml             # PostgreSQL + app orchestration
│   ├── requirements.txt               # Backend dependencies
│   ├── run.py                         # Development server entrypoint
│   └── .env.example                   # Environment variable template
├── predictive-maintenance-pipeline.ipynb  # Training notebook
├── eda-experiments.ipynb              # Exploratory data analysis
├── requirements.txt                   # Root project dependencies
└── README.md                          # This file
```

## Prerequisites

- **Python**: 3.10 or higher
- **pip**: 23.0 or higher
- **Docker** (optional): 24.0 or higher for containerized deployment
- **PostgreSQL** (optional): 15.0 or higher for production database

## Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd machine-learning
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

Root project (training pipeline):
```bash
pip install -r requirements.txt
```

Backend application:
```bash
cd pm-app
pip install -r requirements.txt
```

## Environment Configuration

### Backend Environment Variables

Copy the example configuration:
```bash
cd pm-app
cp .env.example .env
```

Edit `.env` with your settings:

```bash
# Database
DATABASE_URL=sqlite:///pm_app.db
# For PostgreSQL: DATABASE_URL=postgresql://user:password@localhost:5432/pm_database

# Model Storage
MODEL_STORAGE_BACKEND=local
MODEL_STORAGE_PATH=../artifacts

# Gemini AI
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL_NAME=gemini-2.5-pro-exp
GEMINI_STT_MODEL=gemini-2.5-flash
MAX_AUDIO_MB=10

# Flask
FLASK_ENV=development
SECRET_KEY=dev-secret-key-change-in-production
LOG_LEVEL=INFO
CORS_ORIGINS=http://localhost:3000
```

### Required Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABASE_URL` | No | `sqlite:///pm_app.db` | Database connection string |
| `MODEL_STORAGE_PATH` | No | `../artifacts` | Path to trained models |
| `GEMINI_API_KEY` | Yes* | - | Google AI API key (*required for copilot) |
| `GEMINI_MODEL_NAME` | No | `gemini-2.5-pro-exp` | Gemini model for agent LLM |
| `GEMINI_STT_MODEL` | No | `gemini-2.5-flash` | Gemini model for Speech-to-Text |
| `MAX_AUDIO_MB` | No | `10` | Maximum audio file size in MB |
| `SECRET_KEY` | Yes** | - | Flask session secret (**required for production) |

## Running the Backend (pm-app)

### Local Development

1. Initialize database:
```bash
cd pm-app
python scripts/init_db.py
```

2. Start Flask server:
```bash
python run.py
```

Server runs on `http://localhost:5000`

3. Verify health:
```bash
curl http://localhost:5000/health
```

### Docker Deployment

1. Set required environment variables:
```bash
export GEMINI_API_KEY="your_key"
export SECRET_KEY="production_secret"
```

2. Start services:
```bash
cd pm-app
docker-compose up -d
```

3. Check logs:
```bash
docker-compose logs -f app
```

4. Stop services:
```bash
docker-compose down
```

### Docker with PostgreSQL

The `docker-compose.yml` includes PostgreSQL service:

- **Database**: `pm_database`
- **User**: `pm_user`
- **Password**: `pm_password`
- **Port**: `5432`

Schema is automatically initialized via `sql/schema.sql` on first startup.

## Running Machine Learning Scripts

### 1. Dataset Reconstruction

**Purpose**: Create stratified subset with balanced failure/non-failure samples.

**Input**: `data/predictive_maintenance.csv`  
**Output**: `preprocessed/predictive_maintenance_subset_400_machines.csv`

```bash
python scripts/dataset_reconstruction.py
```

**Process**:
- Loads full dataset (10,000 samples)
- Stratifies by failure type and engine type
- Samples 400 machines with distribution preservation
- Validates subset quality

### 2. Time-Series Generation

**Purpose**: Convert static samples to realistic time-series sequences.

**Input**: `preprocessed/predictive_maintenance_subset_400_machines.csv`  
**Output**: `preprocessed/predictive_maintenance_timeseries.csv`

```bash
python scripts/generate_timeseries.py
```

**Process**:
- Generates temporal sequences for each machine
- Adds sensor drift, jitter, and degradation patterns
- Maintains physical constraints (temperature, RPM bounds)
- Creates synthetic RUL decay curves
- Outputs timestamped sequences with ISO 8601 format

**Configuration**: Edit constants in script header:
```python
SENSOR_BOUNDS = {
    'air_temperature_K': (296.0, 304.0),
    'process_temperature_K': (305.0, 314.0),
    'rotational_speed_rpm': (1200, 2500),
    'torque_Nm': (10.0, 70.0),
    'tool_wear_min': (0.0, 250.0)
}
```

### 3. Model Training Pipeline

**Purpose**: Train XGBoost and LSTM models with hyperparameter optimization.

**Input**: `preprocessed/predictive_maintenance_timeseries.csv`  
**Output**: Trained models in `artifacts/`

Run Jupyter notebook:
```bash
jupyter notebook predictive-maintenance-pipeline.ipynb
```

**Training steps**:
1. Load time-series data
2. Feature engineering (lagged features, rolling windows)
3. Train/test split (80/20)
4. Hyperparameter tuning with Optuna (50 trials)
5. Train XGBoost classifier and regressor
6. Train LSTM models (optional)
7. Export models to `artifacts/`
8. Generate performance metrics

**Expected outputs**:
```
artifacts/
├── xgb_classifier.joblib          # Accuracy: ~98.75%
├── xgb_regressor.joblib           # RMSE: ~8.2 steps
├── scaler.joblib                  # StandardScaler
├── encoder_engine_type.joblib     # LabelEncoder
├── feature_importance_*.csv       # Feature rankings
└── metrics_report.txt             # Performance summary
```

## Database Management

### Initialize Database

Create tables from schema:
```bash
cd pm-app
python scripts/init_db.py
```

With custom database URL:
```bash
python scripts/init_db.py --database-url postgresql://user:pass@host/db
```

Drop existing tables before creating:
```bash
python scripts/init_db.py --drop-existing
```

### Add Conversation History Table

Create the `conversation_history` table for LangChain copilot memory:
```bash
python scripts/add_conversation_history_table.py
```

Force recreate if table exists:
```bash
python scripts/add_conversation_history_table.py --force
```

### Schema Migration

Migrate existing data to legacy schema format:
```bash
cd pm-app
python scripts/migrate_to_legacy_schema.py
```

Preview changes without applying:
```bash
python scripts/migrate_to_legacy_schema.py --dry-run
```

With custom database URL:
```bash
python scripts/migrate_to_legacy_schema.py --database-url postgresql://user:pass@host/db
```

**Migration details**:
- Moves `path` and `metrics` columns into `metadata` JSONB field
- Supports both PostgreSQL and SQLite
- Creates backup before migration
- Validates data integrity

## API Endpoints

### Health Check

```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "database": "connected",
  "models_loaded": ["xgb_classifier", "xgb_regressor"]
}
```

### Failure Classification

Predict equipment failure probability.

```bash
POST /predict/classification
Content-Type: application/json

{
  "model_name": "xgb_classifier",
  "product_id": "L56614",
  "unit_id": "9435",
  "timestamp_before": "2024-01-15T10:00:00Z"
}
```

Response:
```json
{
  "data": {
    "failure_probability": 0.87,
    "predicted_class": 1,
    "confidence": 0.92,
    "features_used": {...}
  },
  "error": null
}
```

### RUL Forecasting

Predict remaining useful life in time steps.

```bash
POST /predict/forecast
Content-Type: application/json

{
  "model_name": "xgb_regressor",
  "product_id": "PROD_001",
  "unit_id": "9435",
  "horizon_steps": 10
}
```

Response:
```json
{
  "data": {
    "predicted_rul": 45.2,
    "unit": "steps",
    "confidence_interval": [40.1, 50.3]
  },
  "error": null
}
```

### Maintenance Scheduling

Generate optimized maintenance schedule.

```bash
POST /optimizer/schedule
Content-Type: application/json

{
  "unit_list": [
    {"product_id": "PROD_001", "unit_id": "UNIT_001"},
    {"product_id": "PROD_001", "unit_id": "UNIT_002"}
  ],
  "risk_threshold": 0.7,
  "rul_threshold": 24.0,
  "horizon_days": 7,
  "teams_available": 2
}
```

Response:
```json
{
  "data": {
    "schedule": [
      {
        "unit_id": "9435",
        "recommended_start": "2024-01-16T08:00:00Z",
        "recommended_end": "2024-01-16T12:00:00Z",
        "reason": "High failure risk: 0.87",
        "priority": 1
      }
    ],
    "summary": {
      "total_units": 2,
      "scheduled": 1,
      "deferred": 1
    }
  },
  "error": null
}
```

### Model Management

List loaded models:
```bash
GET /models
```

Reload models from disk:
```bash
POST /models/reload
```

Trigger retraining:
```bash
POST /retrain/run
Content-Type: application/json

{
  "model_type": "classification",
  "incremental": true
}
```

### Copilot Chat

Natural language maintenance queries powered by LangChain agent with Gemini LLM.

```bash
POST /copilot/chat
Content-Type: application/json
X-Session-ID: user_123  # Optional header for session tracking

{
  "messages": [
    {"role": "user", "content": "What's the failure risk for unit L56614/9435?"}
  ],
  "session_id": "user_123",
  "context": {"product_id": "L56614"}
}
```

Response:
```json
{
  "data": {
    "reply": "Unit L56614/9435 has a 4.45% failure probability with high confidence. Equipment operating normally.",
    "session_id": "user_123",
    "execution_time_seconds": 2.34,
    "timestamp": "2025-12-02T00:40:08.618Z",
    "intermediate_steps": [
      {
        "tool": "predict_failure",
        "tool_input": {"product_id": "L56614", "unit_id": "9435"},
        "output": "{success: true, failure_probability: 0.0445}"
      }
    ]
  },
  "error": null
}
```

**Clear session history:**
```bash
DELETE /copilot/session/{session_id}
```

**Supported queries:**
- "What's the failure risk for L56614/9435?"
- "Check unit M29501/1234"
- "How much RUL does unit H30221/5678 have?"
- "Schedule maintenance for units L56614, M29501, H30221"
- "Which units need immediate attention?"
- "Show all units status" (triggers global risk assessment)

### Copilot Voice Input

Natural language voice queries via Speech-to-Text powered by Gemini.

```bash
POST /copilot/voice
Content-Type: multipart/form-data

audio: <binary audio file>
session_id: user_123  # Optional
language: id          # Optional (default: id for Indonesian)
```

**Supported audio formats:**
- `audio/webm`
- `audio/wav`, `audio/x-wav`
- `audio/mp3`, `audio/mpeg`
- `audio/ogg`

**File size limit:** 10MB (configurable via `MAX_AUDIO_MB`)

Response:
```json
{
  "data": {
    "transcription": "Berapa risiko kegagalan unit L56614/9435?",
    "reply": "Unit L56614/9435 memiliki probabilitas kegagalan 4.45% dengan kepercayaan tinggi.",
    "session_id": "user_123",
    "execution_time_seconds": 3.21,
    "timestamp": "2025-12-02T01:15:00.000Z",
    "intermediate_steps": [...]
  },
  "error": null
}
```

**Error responses:**
- 400: Missing audio file, unsupported format, or file too large
- 500: Transcription or agent processing error

### Maintenance Schedule Records

Retrieve persisted maintenance schedules from database.

```bash
GET /maintenance/schedule?product_id=L56614&status=pending
```

**Query parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `product_id` | string | Filter by product ID |
| `unit_id` | string | Filter by unit ID |
| `status` | string | Filter by status (pending, completed, cancelled) |
| `start_date` | string | ISO date for scheduled_start >= |
| `end_date` | string | ISO date for scheduled_start <= |

Response:
```json
{
  "data": [
    {
      "id": 1,
      "product_id": "L56614",
      "unit_id": "9435",
      "scheduled_start": "2025-01-20T08:00:00Z",
      "scheduled_end": "2025-01-20T12:00:00Z",
      "status": "pending",
      "reason": "High failure risk: 0.87",
      "priority": 1,
      "assigned_team": "Team A",
      "created_at": "2025-01-15T10:00:00Z"
    }
  ],
  "error": null
}
```

### Model Retraining

Trigger model retraining with optional incremental learning.

```bash
POST /retrain/run
Content-Type: application/json

{
  "model_type": "classification",
  "incremental": true,
  "sample_limit": 5000
}
```

**Body parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_type` | string | classification | Type: classification or regression |
| `incremental` | boolean | false | Continue from existing model weights |
| `sample_limit` | integer | null | Limit training samples |

Response:
```json
{
  "data": {
    "model_type": "classification",
    "status": "started",
    "timestamp": "2025-01-15T10:00:00Z",
    "message": "Retraining started for classification"
  },
  "error": null
}
```

**Key files:**
- `langchain_agent_service.py` - Agent orchestration with PostgreSQL-backed memory
- `agent_tools.py` - LangChain tools with Pydantic input validation
- `transcription_service.py` - Gemini STT for voice input
- `copilot_voice.py` - Voice endpoint with audio validation
- `copilot_service.py` - Direct Gemini API client (legacy)

## Testing

### Run All Tests

```bash
cd pm-app
pytest tests/ -v
```

### Run Specific Test Suite

```bash
pytest tests/test_classification_api.py -v
pytest tests/test_forecast_api.py -v
pytest tests/test_optimizer_api.py -v
```

### Run with Coverage

```bash
pytest tests/ --cov=app --cov-report=html
```

View coverage report:
```bash
open htmlcov/index.html  # macOS/Linux
start htmlcov/index.html # Windows
```

### Test Database

Tests use ephemeral SQLite database with automatic cleanup. No manual database setup required.

**Test fixtures** (`tests/conftest.py`):
- `test_db`: Temporary database engine
- `db_session`: Isolated database session per test
- `app`: Flask test application
- `client`: Flask test client
- `sample_telemetry`: Mock telemetry data
- `sample_model_artifact`: Mock model metadata

## Model Artifacts

### Storage Layout

```
artifacts/
├── xgb_classifier.joblib          # Binary classification model
├── xgb_regressor.joblib           # RUL regression model
├── lstm_classifier.keras          # Alternative LSTM classifier
├── lstm_regressor.keras           # Alternative LSTM regressor
├── scaler.joblib                  # StandardScaler (fit on training data)
├── encoder_engine_type.joblib     # LabelEncoder for categorical features
├── feature_importance_classifier.csv
├── feature_importance_regressor.csv
├── metrics_report.txt
└── inference_results.csv
```

### Model Registry

Models are registered in `model_artifact` table:

```sql
INSERT INTO model_artifact (model_name, version, metadata, promoted_at)
VALUES (
  'xgb_classifier',
  '20240115_120000',
  '{"path": "file:///app/artifacts/xgb_classifier.joblib", 
    "metrics": {"accuracy": 0.9875, "f1_score": 0.9821},
    "hyperparameters": {"n_estimators": 150, "max_depth": 8}}',
  NOW()
);
```

### Model Loading

Backend loads models via convention-based paths:

1. Check `metadata.path` in `model_artifact` table
2. Fallback to `{model_name}_{version}.joblib`
3. Support `file://` and `gs://` URI schemes
4. Cache loaded models in memory

## Dataset Workflow

### 1. Load Raw Data

```python
import pandas as pd
df = pd.read_csv('data/predictive_maintenance.csv')
```

### 2. Create Stratified Subset

```bash
python scripts/dataset_reconstruction.py
```

Output: `preprocessed/predictive_maintenance_subset_400_machines.csv`

### 3. Generate Time-Series

```bash
python scripts/generate_timeseries.py
```

Output: `preprocessed/predictive_maintenance_timeseries.csv`

### 4. Train Models

```bash
jupyter notebook predictive-maintenance-pipeline.ipynb
```

Run all cells to train and export models.

### 5. Load Data into Database

```python
import pandas as pd
from sqlalchemy import create_engine

df = pd.read_csv('preprocessed/predictive_maintenance_timeseries.csv')
engine = create_engine('postgresql://user:pass@host/db')
df.to_sql('telemetry', engine, if_exists='append', index=False)
```

### 6. Verify Ingestion

```bash
curl -X POST http://localhost:5000/predict/classification \
  -H "Content-Type: application/json" \
  -d '{"model_name": "xgb_classifier", "product_id": "PROD_001", "unit_id": "9435"}'
```

## Deployment Notes

### Docker Production

Build production image:
```bash
cd pm-app
docker build -t pm-app:latest --target production .
```

Run with mounted artifacts:
```bash
docker run -d \
  -p 5000:5000 \
  -v $(pwd)/../artifacts:/app/artifacts:ro \
  -e DATABASE_URL=postgresql://user:pass@host/db \
  -e GEMINI_API_KEY=your_key \
  -e SECRET_KEY=production_secret \
  pm-app:latest
```

### Environment Best Practices

- **Development**: Use SQLite with `DATABASE_URL=sqlite:///pm_app.db`
- **Staging**: Use PostgreSQL with connection pooling
- **Production**: Use PostgreSQL with separate credentials, TLS connections

### Model Artifacts in Docker

Mount artifacts as read-only volume:
```yaml
volumes:
  - ./artifacts:/app/artifacts:ro
```

Or copy into image during build:
```dockerfile
COPY ../artifacts /app/artifacts
```

### Database Migrations

For schema changes:
1. Update `app/models.py`
2. Create migration script in `pm-app/scripts/`
3. Test migration on staging database
4. Apply to production with downtime window

## Troubleshooting

### Missing Dependencies

**Error**: `ModuleNotFoundError: No module named 'flask'`

**Solution**:
```bash
cd pm-app
pip install -r requirements.txt
```

### Missing Model Artifacts

**Error**: `FileNotFoundError: Model file not found: artifacts/xgb_classifier.joblib`

**Solution**:
1. Train models: `jupyter notebook predictive-maintenance-pipeline.ipynb`
2. Verify artifacts exist: `ls artifacts/xgb_*.joblib`
3. Check `MODEL_STORAGE_PATH` in `.env`

### Database Connection Issues

**Error**: `sqlalchemy.exc.OperationalError: could not connect to server`

**Solution**:

For PostgreSQL:
```bash
# Check PostgreSQL is running
pg_isready -h localhost -p 5432

# Verify credentials
psql -U pm_user -d pm_database -h localhost
```

For SQLite:
```bash
# Verify file permissions
ls -l pm_app.db

# Recreate database
rm pm_app.db
python scripts/init_db.py
```

### Script Execution Errors

**Error**: `FileNotFoundError: data/predictive_maintenance.csv`

**Solution**:
```bash
# Verify data files exist
ls data/*.csv

# Download missing datasets from source
```

**Error**: `ValueError: Timestamp format not recognized`

**Solution**:

Ensure timestamps are ISO 8601 format:
```python
# Correct format
"2024-01-15T10:00:00Z"
"2024-01-15T10:00:00+00:00"

# Fix in pandas
df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%dT%H:%M:%S%z')
```

### Docker Issues

**Error**: `port 5000 already in use`

**Solution**:
```bash
# Find process using port
lsof -i :5000  # macOS/Linux
netstat -ano | findstr :5000  # Windows

# Kill process or change port in docker-compose.yml
```

**Error**: `ERROR: Cannot connect to the Docker daemon`

**Solution**:
```bash
# Start Docker service
sudo systemctl start docker  # Linux
# Or start Docker Desktop application
```

### Test Failures

**Error**: `AssertionError: assert 500 == 200`

**Solution**:
```bash
# Run tests with verbose output
pytest tests/test_classification_api.py -vv -s

# Check test logs for error details
# Verify test database is clean
rm -f test_*.db
```

### Gemini API Issues

**Error**: `401 Unauthorized: Invalid API key`

**Solution**:
```bash
# Verify API key is set
echo $GEMINI_API_KEY

# Set in .env file
echo "GEMINI_API_KEY=your_key_here" >> pm-app/.env
```

**Error**: `429 Too Many Requests: Rate limit exceeded`

**Solution**:
- Reduce request frequency
- Implement exponential backoff
- Upgrade Gemini API quota

## Performance Tuning

### Database Optimization

```sql
-- Add indexes for common queries
CREATE INDEX idx_telemetry_lookup ON telemetry(product_id, unit_id, timestamp);
CREATE INDEX idx_model_artifact_latest ON model_artifact(model_name, promoted_at DESC);

-- Vacuum and analyze
VACUUM ANALYZE telemetry;
```

### Model Caching

Models are cached in memory on first load. To preload:
```python
from app.services.models_loader import get_models_loader
loader = get_models_loader()
loader.get_classification_model('xgb_classifier')
loader.get_forecast_model('xgb_regressor')
```

### Connection Pooling

Configure in `.env`:
```bash
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
DB_POOL_TIMEOUT=30
```
