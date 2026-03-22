# embedding_manager.py

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingManager:
  
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        
        self.model_name = model_name
        self.model = None
        self.load_model()
    
    def load_model(self) -> None:
       
        try:
            self.model = SentenceTransformer(self.model_name)
            embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"Loaded embedding model: {self.model_name}. Embedding dimension: {embedding_dim}")
        except Exception as e:
            print(f"Error loading model '{self.model_name}': {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
       
        if not self.model:
            raise ValueError("Embedding model not loaded.")
        try:
            embeddings = self.model.encode(texts, show_progress_bar=True)
            return np.array(embeddings)
        except Exception as e:
            print(f"Error during embedding: {e}")
            raise

    def get_embedding_dimension(self) -> int:
       
        if not self.model:
            raise ValueError("Embedding model not loaded.")
        return self.model.get_sentence_embedding_dimension()
