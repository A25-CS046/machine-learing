import logging
import os
import io  # Added for GCS byte stream
from datetime import datetime, timezone
from typing import Literal
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sqlalchemy import and_, text
from app.db import get_db
from app.models import Telemetry, RetrainPointer, ModelArtifact
from app.config import load_config
from app.services.classification_service import FEATURE_COLUMNS, aggregate_features_from_telemetry

logger = logging.getLogger(__name__)

def load_encoder(config):
    """
    Helper function to load the encoder from GCS or Local storage.
    """
    filename = 'encoder_engine_type.joblib'
    
    # 1. GCS Loading
    if config.model_storage.backend == 'gcs':
        try:
            from google.cloud import storage
            bucket_name = config.model_storage.gcs_bucket
            if not bucket_name:
                raise ValueError("GCS_MODEL_BUCKET not set")
            
            logger.info(f"Downloading encoder from GCS: gs://{bucket_name}/{filename}")
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(filename)
            
            if not blob.exists():
                raise FileNotFoundError(f"Encoder not found in GCS: {filename}")
                
            blob_bytes = blob.download_as_bytes()
            return joblib.load(io.BytesIO(blob_bytes))
            
        except ImportError:
            logger.error("google-cloud-storage library missing")
            raise
        except Exception as e:
            logger.error(f"Failed to load encoder from GCS: {e}")
            raise

    # 2. Local Loading (Fallback)
    else:
        local_path = os.path.join(config.model_storage.local_path, filename)
        if not os.path.exists(local_path):
             # Fallback check for common deployment locations if default fails
             base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
             candidates = [
                 os.path.join(base_dir, 'artifacts', filename),
                 os.path.join(base_dir, 'pm-app', 'artifacts', filename)
             ]
             for path in candidates:
                 if os.path.exists(path):
                     local_path = path
                     break
        
        logger.info(f"Loading encoder from local path: {local_path}")
        return joblib.load(local_path)


def retrain_model(
    model_type: Literal['classification', 'forecast'],
    incremental: bool = True,
    sample_limit: int | None = None
) -> dict:
    """
    Retrain XGBoost model using telemetry data.
    Supports incremental training from last retrain checkpoint.
    """
    config = load_config()
    model_name = 'xgb_classifier' if model_type == 'classification' else 'xgb_regressor'
    
    with get_db() as session:
        pointer = session.query(RetrainPointer).filter(
            RetrainPointer.model_name == model_name
        ).first()
        
        if pointer is None:
            pointer = RetrainPointer(
                model_name=model_name,
                last_retrain_ts=datetime(1970, 1, 1, tzinfo=timezone.utc),
                last_retrain_id=0
            )
            session.add(pointer)
            session.commit()
        
        last_retrain_ts = pointer.last_retrain_ts
        last_retrain_ts_str = last_retrain_ts.isoformat()
        
        if incremental:
            query = session.query(Telemetry).filter(
                text("timestamp::timestamptz > :last_ts")
            ).params(last_ts=last_retrain_ts_str)
        else:
            query = session.query(Telemetry)
        
        if model_type == 'forecast':
            query = query.filter(Telemetry.synthetic_RUL.isnot(None))
        
        query = query.order_by(text("timestamp::timestamptz"))
        
        if sample_limit:
            query = query.limit(sample_limit)
        
        telemetry_data = query.all()
        
        if not telemetry_data:
            raise ValueError("No new telemetry data available for retraining")
        
        logger.info(f"Found {len(telemetry_data)} rows for retraining {model_name}")
        
        units = {}
        for row in telemetry_data:
            key = (row.product_id, row.unit_id)
            if key not in units:
                units[key] = []
            units[key].append(row)
        
        # --- FIX: Use the new helper function to load encoder ---
        try:
            encoder = load_encoder(config)
        except Exception as e:
            logger.error(f"Preprocessing artifacts not found: {e}")
            raise ValueError(f"Encoder not found: {e}")
        # ------------------------------------------------------
        
        X_list = []
        y_list = []
        
        for unit_rows in units.values():
            if len(unit_rows) < 5:
                continue
            
            feature_vector = aggregate_features_from_telemetry(unit_rows, encoder)
            
            if model_type == 'classification':
                label = 1 if any(r.is_failure == 1 for r in unit_rows) else 0
            else:
                label = unit_rows[-1].synthetic_RUL if unit_rows[-1].synthetic_RUL else 0.0
            
            X_list.append(feature_vector[0])
            y_list.append(label)
        
        if not X_list:
             raise ValueError("Not enough data points after filtering (need 5+ records per unit)")

        X = np.array(X_list)
        y = np.array(y_list)
        
        logger.info(f"Training dataset: {X.shape[0]} samples, {X.shape[1]} features")
        
        if model_type == 'classification':
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        
        model.fit(X, y)
        
        version = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        model_filename = f"{model_name}_{version}.joblib"

        # --- UPDATE: Save Trained Model to GCS or Local ---
        if config.model_storage.backend == 'gcs':
            from google.cloud import storage
            bucket_name = config.model_storage.gcs_bucket
            
            # Save to local buffer first
            buffer = io.BytesIO()
            joblib.dump(model, buffer)
            buffer.seek(0)
            
            # Upload to GCS
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(model_filename)
            blob.upload_from_file(buffer)
            
            model_uri = f"gs://{bucket_name}/{model_filename}"
            logger.info(f"Uploaded retrained model to {model_uri}")
            
        else:
            model_path = os.path.join(config.model_storage.local_path, model_filename)
            joblib.dump(model, model_path)
            model_uri = f"file://{model_path}"
        # --------------------------------------------------
        
        if model_type == 'classification':
            from sklearn.metrics import accuracy_score, recall_score
            y_pred = model.predict(X)
            metrics = {
                'accuracy': float(accuracy_score(y, y_pred)),
                'recall': float(recall_score(y, y_pred, zero_division=0)),
                'samples': int(len(y))
            }
        else:
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            y_pred = model.predict(X)
            metrics = {
                'mae': float(mean_absolute_error(y, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
                'r2': float(r2_score(y, y_pred)),
                'samples': int(len(y))
            }
        
        artifact = ModelArtifact(
            model_name=model_name,
            version=version,
            model_metadata={
                'path': model_uri,
                'metrics': metrics,
                'model_type': model_type,
                'incremental': incremental,
                'training_samples': len(y)
            }
        )
        session.add(artifact)
        
        last_row = telemetry_data[-1]
        last_timestamp_str = last_row.timestamp
        try:
            last_timestamp_dt = datetime.fromisoformat(last_timestamp_str.replace('Z', '+00:00'))
            if last_timestamp_dt.tzinfo is None:
                last_timestamp_dt = last_timestamp_dt.replace(tzinfo=timezone.utc)
        except (ValueError, AttributeError):
            last_timestamp_dt = datetime.now(timezone.utc)
        
        pointer.last_retrain_ts = last_timestamp_dt
        pointer.last_retrain_id = 0
        
        session.commit()
        
        logger.info(f"Model {model_name} v{version} trained and promoted")
        
        return {
            'model_name': model_name,
            'version': version,
            'path': model_uri,
            'metrics': metrics,
            'training_samples': len(y),
            'incremental': incremental
        }