# embedding_manager.py

from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import time
from src.logging_config import setup_logger

logger = setup_logger(__name__)


class EmbeddingManager:
  
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        
        self.model_name = model_name
        self.model = None
        self.load_model()
    
    def load_model(self) -> None:
       
        try:
            self.model = SentenceTransformer(self.model_name)
            embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded embedding model: {self.model_name} | Dimension: {embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load model '{self.model_name}': {e}")
            raise

    def generate_embeddings(self, texts: List[str], max_retries: int = 2) -> np.ndarray:
        
        if not self.model:
            raise ValueError("Embedding model not loaded.")
        
        for attempt in range(max_retries + 1):
            try:
                embeddings = self.model.encode(texts, show_progress_bar=True)
                return np.array(embeddings)
            except Exception as e:
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(f"Embedding error (attempt {attempt + 1}/{max_retries + 1}): {str(e)[:100]}")
                    logger.debug(f"Retrying embed in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Embedding generation failed after {max_retries + 1} attempts: {e}")
                    raise RuntimeError(f"Embedding generation failed: {str(e)}")

    def get_embedding_dimension(self) -> int:
       
        if not self.model:
            raise ValueError("Embedding model not loaded.")
        return self.model.get_sentence_embedding_dimension()
