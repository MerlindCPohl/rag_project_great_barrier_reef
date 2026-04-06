"""
Embedding Manager for generating and managing text embeddings.
Uses SentenceTransformers (BAAI/bge-m3) to convert text documents into vector
embeddings for similarity search in the RAG pipeline.
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from src.utils import setup_logger

logger = setup_logger(__name__)


class EmbeddingManager:
    """
    Manages conversion from text to embeddings using sentence transformer.
    Loads the model, generates embeddings and gets their dimensions. 
    """
  
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        
        self.model_name = model_name
        self.model = None
        self.load_model()
    
    def load_model(self) -> None:
       
        try:
            self.model = SentenceTransformer(self.model_name)
            embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded embedding model: {self.model_name}. Embedding dimension: {embedding_dim}")
        except Exception as e:
            logger.error(f"Error loading model '{self.model_name}': {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        
        if not self.model:
            raise ValueError("Embedding model not loaded.")
        
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return np.array(embeddings)

    def get_embedding_dimension(self) -> int:
       
        if not self.model:
            raise ValueError("Embedding model not loaded.")
        return self.model.get_sentence_embedding_dimension()
