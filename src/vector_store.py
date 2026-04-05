#vector storing and searching using FAISS

import os
import faiss
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from src.logging_config import setup_logger

logger = setup_logger(__name__)

class FaissVectorStore:

    def __init__(self, embedding_dim: int, persist_directory: Optional[str] = None) -> None:
        self.embedding_dim = embedding_dim

        if persist_directory is None:
            base_path = os.path.dirname(os.path.abspath(__file__))
            self.persist_directory = os.path.join(base_path, "..", "data", "vector_store")
        else:
            self.persist_directory = persist_directory
            
        self.index_path = os.path.join(self.persist_directory, "index.faiss")
        self.metadata_path = os.path.join(self.persist_directory, "metadata.pkl")
        
        # if exists skips the creation, otherwise creates the directory
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # load index if exists, otherwise create new
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            # Verify dimension compatibility
            if self.index.d != embedding_dim:
                logger.warning(f"WARNING: Existing index has dimension {self.index.d}, but {embedding_dim} was provided.")
                logger.info(f"Recreating index with dimension {embedding_dim}...")
                self.index = faiss.IndexFlatIP(embedding_dim)
                self.id_to_metadata = {}
                logger.info("New index created")
            else:
                with open(self.metadata_path, "rb") as f:
                    self.id_to_metadata = pickle.load(f)
                logger.info(f"Loaded index with {len(self.id_to_metadata)} chunks")
        else:
            self.index = faiss.IndexFlatIP(embedding_dim) # IP = inner Product, equivalent to cosine similarity
            self.id_to_metadata = {}
            logger.info("Created new vector index")


    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        
        embeddings = np.array(embeddings)
        
        # If 1D array, reshape to 2D
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        # Compute L2 norm for each embedding (row-wise)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms
    

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]) -> None:
        embeddings = self.normalize_embeddings(embeddings)
        
        # Validate dimensions before adding
        if embeddings.shape[1] != self.index.d:
            raise ValueError(
                f"Embedding dimension mismatch: embeddings have dimension {embeddings.shape[1]}, "
                f"but index expects dimension {self.index.d}"
            )
        
        self.index.add(embeddings.astype('float32'))
        
        # conncet metadata with IDs
        current_size = len(self.id_to_metadata)
        for i, meta in enumerate(metadatas):
            self.id_to_metadata[current_size + i] = meta
            
        # persistently save to disk
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.id_to_metadata, f)
        logger.debug(f"Saved {len(metadatas)} chunks to vector store ({len(self.id_to_metadata)} total)")
        

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        
        query_embedding = self.normalize_embeddings(query_embedding)
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        results = [(self.id_to_metadata[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
        return results