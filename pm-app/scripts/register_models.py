"""Register trained models in the database (supports GCS and Local)."""
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.models import ModelArtifact, Base

# Import Google Cloud Storage
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

def upload_to_gcs(local_path, bucket_name, destination_blob_name):
    """Uploads a file to the bucket."""
    if not GCS_AVAILABLE:
        raise ImportError("google-cloud-storage is required for GCS upload. Run: pip install google-cloud-storage")
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    print(f"Uploading {local_path} to gs://{bucket_name}/{destination_blob_name}...")
    blob.upload_from_filename(local_path)
    print(f"Upload complete.")
    
    return f"gs://{bucket_name}/{destination_blob_name}"

def register_models():
    """Register trained models from artifacts directory."""
    
    # 1. Configuration
    db_url = os.getenv('DATABASE_URL')
    storage_backend = os.getenv('MODEL_STORAGE_BACKEND', 'local')
    gcs_bucket = os.getenv('GCS_MODEL_BUCKET')

    if not db_url:
        print("ERROR: DATABASE_URL not found.")
        sys.exit(1)

    # Validate GCS config
    if storage_backend == 'gcs' and not gcs_bucket:
        print("ERROR: MODEL_STORAGE_BACKEND is 'gcs' but GCS_MODEL_BUCKET is not set.")
        sys.exit(1)

    print(f"Connecting to database...")
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    artifacts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../artifacts'))
    
    # Model Definitions
    models_to_register = [
        {
            'model_name': 'xgb_classifier',
            'version': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'filename': 'xgb_classifier.joblib',
            'metrics': {'accuracy': 0.9875, 'f1_score': 0.9827}
        },
        {
            'model_name': 'xgb_regressor',
            'version': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'filename': 'xgb_regressor.joblib',
            'metrics': {'rmse': 8.2, 'r2_score': 0.92}
        },
        {
            'model_name': 'scaler',
            'version': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'filename': 'scaler.joblib',
            'metrics': {'type': 'preprocessor', 'method': 'StandardScaler'}
        },
        {
            'model_name': 'encoder_engine_type',
            'version': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'filename': 'encoder_engine_type.joblib',
            'metrics': {'type': 'preprocessor', 'method': 'LabelEncoder'}
        },
    ]
    
    try:
        registered_count = 0
        for model_info in models_to_register:
            local_file_path = os.path.join(artifacts_path, model_info['filename'])
            
            # Check if local file exists before trying anything
            if not os.path.exists(local_file_path):
                print(f"⚠ Warning: Local model file not found: {local_file_path}")
                continue
            
            # 2. Determine URI based on Backend
            final_uri = ""
            
            if storage_backend == 'gcs':
                # Upload to GCS and get gs:// URI
                try:
                    final_uri = upload_to_gcs(
                        local_file_path, 
                        gcs_bucket, 
                        model_info['filename']
                    )
                except Exception as e:
                    print(f"Failed to upload {model_info['filename']} to GCS: {e}")
                    continue
            else:
                # Use Local File URI
                final_uri = f"file://{local_file_path.replace(os.sep, '/')}"

            # 3. Register in Database
            artifact = ModelArtifact(
                model_name=model_info['model_name'],
                version=model_info['version'],
                model_metadata={
                    'path': final_uri,  # This now stores the correct gs:// or file:// URI
                    'metrics': model_info['metrics'],
                    'storage_backend': storage_backend,
                    'registered_at': datetime.now(timezone.utc).isoformat()
                },
                promoted_at=datetime.now(timezone.utc)
            )
            
            session.add(artifact)
            print(f"✓ Registered {model_info['model_name']} at {final_uri}")
            registered_count += 1
        
        session.commit()
        print(f"\n✓ Successfully registered {registered_count} models")
        
    except Exception as e:
        session.rollback()
        print(f"✗ Registration failed: {e}")
        raise
    finally:
        session.close()

if __name__ == '__main__':
    register_models()