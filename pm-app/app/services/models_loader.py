import io
import logging
import os
import pathlib
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname
import joblib
from sqlalchemy import desc
from app.config import load_config
from app.db import get_db
from app.models import ModelArtifact

logger = logging.getLogger(__name__)

class ModelsLoader:
    def __init__(self):
        self._cache: dict[str, Any] = {}
        self._config = load_config()
        # Calculate absolute project root (pm-app)
        self._base_dir = pathlib.Path(__file__).resolve().parent.parent.parent

    def _parse_uri(self, uri: str) -> tuple[str, str, str]:
        """
        Parses URI and returns (scheme, netloc, path).
        For 'gs://my-bucket/model.joblib':
          scheme='gs', netloc='my-bucket', path='/model.joblib'
        """
        parsed = urlparse(uri)
        return parsed.scheme, parsed.netloc, parsed.path
    
    def _load_from_local(self, path_from_uri: str) -> Any:
        """
        Robust local loader that handles Windows paths and fallbacks.
        """
        # 1. Convert URI path to system path
        clean_path = url2pathname(path_from_uri)
        
        # 2. Primary attempt
        if os.path.exists(clean_path):
            return joblib.load(clean_path)

        # 3. Fallback Strategy
        filename = os.path.basename(clean_path)
        candidates = [
            clean_path,                                  
            self._base_dir.parent / 'artifacts' / filename, 
            self._base_dir / 'artifacts' / filename,        
            self._base_dir / filename,                      
            self._base_dir / 'app' / 'artifacts' / filename 
        ]

        for candidate in candidates:
            candidate_str = str(candidate)
            if os.path.exists(candidate_str):
                logger.info(f"Found model file via fallback at: {candidate_str}")
                return joblib.load(candidate_str)

        # 4. Debugging
        raise FileNotFoundError(f"Model file not found: {clean_path}")
    
    def _load_from_gcs(self, bucket_name: str, blob_path: str) -> Any:
        """
        Loads a model directly from GCS into memory.
        """
        try:
            from google.cloud import storage
        except ImportError:
            raise ImportError("google-cloud-storage not installed.")
        
        # Fallback to config bucket if URI didn't have one (rare for valid gs:// URIs)
        if not bucket_name:
            bucket_name = self._config.model_storage.gcs_bucket
            if not bucket_name:
                raise ValueError("No bucket specified in URI and GCS_MODEL_BUCKET not set")

        # Clean up the path (remove leading slash if present)
        blob_path = blob_path.lstrip('/')
        
        logger.info(f"Downloading from GCS: gs://{bucket_name}/{blob_path}")

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        
        if not blob.exists():
             raise FileNotFoundError(f"GCS Object not found: gs://{bucket_name}/{blob_path}")

        # Download to memory file-like object
        blob_bytes = blob.download_as_bytes()
        return joblib.load(io.BytesIO(blob_bytes))
    
    def load_model_from_uri(self, uri: str) -> Any:
        if uri in self._cache:
            logger.debug(f"Model cache hit: {uri}")
            return self._cache[uri]
        
        scheme, netloc, path = self._parse_uri(uri)
        
        logger.info(f"Loading model from URI: {uri}")
        
        if scheme == 'file' or not scheme:
            model = self._load_from_local(path)
        elif scheme == 'gs':
            # netloc is the bucket, path is the blob path
            model = self._load_from_gcs(netloc, path)
        else:
            raise ValueError(f"Unsupported model URI scheme: {scheme}")
        
        self._cache[uri] = model
        logger.info(f"Model loaded and cached: {uri}")
        return model
    
    def _build_model_path(self, model_name: str, version: str, metadata: dict | None = None) -> str:
        # Priority 1: Use the specific path stored in DB metadata (Preferred for Production)
        if metadata and 'path' in metadata:
            return metadata['path']
        
        # Priority 2: Construct path based on environment config (Fallback)
        model_filename = f"{model_name}_{version}.joblib"
        
        if self._config.model_storage.backend == 'local':
            full_path = os.path.join(self._config.model_storage.local_path, model_filename)
            return pathlib.Path(os.path.abspath(full_path)).as_uri()
        elif self._config.model_storage.backend == 'gcs':
            return f"gs://{self._config.model_storage.gcs_bucket}/{model_filename}"
        else:
            raise ValueError(f"Unknown storage backend: {self._config.model_storage.backend}")
    
    def get_latest_model_artifact(self, model_name: str) -> ModelArtifact:
        with get_db() as session:
            artifact = session.query(ModelArtifact).filter(
                ModelArtifact.model_name == model_name
            ).order_by(desc(ModelArtifact.promoted_at)).first()
            
            if not artifact:
                raise ValueError(f"No model artifact found for: {model_name}")
            
            # Detach from session
            session.expunge(artifact)
            return artifact
    
    def get_model(self, model_name: str) -> tuple[Any, ModelArtifact]:
        artifact = self.get_latest_model_artifact(model_name)
        model_path = self._build_model_path(
            artifact.model_name,
            artifact.version,
            artifact.model_metadata
        )
        model = self.load_model_from_uri(model_path)
        return model, artifact
    
    def get_classification_model(self, model_name: str = 'xgb_classifier') -> tuple[Any, ModelArtifact]:
        return self.get_model(model_name)
    
    def get_forecast_model(self, model_name: str = 'xgb_regressor') -> tuple[Any, ModelArtifact]:
        return self.get_model(model_name)
    
    # --- FIXED METHODS BELOW ---
    
    def get_scaler(self) -> Any:
        """
        Loads the scaler using the database record instead of a hardcoded local path.
        """
        # This will now fetch the 'gs://' path from the DB
        model, _ = self.get_model('scaler')
        return model
    
    def get_encoder(self) -> Any:
        """
        Loads the encoder using the database record instead of a hardcoded local path.
        """
        # This will now fetch the 'gs://' path from the DB
        model, _ = self.get_model('encoder_engine_type')
        return model
    
    def reload_all(self) -> None:
        logger.info("Clearing model cache")
        self._cache.clear()

    def get_cached_models(self) -> list[str]:
        return list(self._cache.keys())


_loader_instance: ModelsLoader | None = None

def get_models_loader() -> ModelsLoader:
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = ModelsLoader()
    return _loader_instance