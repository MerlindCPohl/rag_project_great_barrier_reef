#vector storing and searching using FAISS

import os
import faiss
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple

class FaissVectorStore:

    def __init__(self, embedding_dim: int, persist_directory: str = None):
        self.embedding_dim = embedding_dim

        if persist_directory is None:
            base_path = os.path.dirname(os.path.abspath("__file__"))
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
            with open(self.metadata_path, "rb") as f:
                self.id_to_metadata = pickle.load(f)
            print(f"Loaded index with{len(self.id_to_metadata)} Chunks")
        else:
            self.index = faiss.IndexFlatIP(embedding_dim) # IP = inner Product, equivalent to cosine similarity
            self.id_to_metadata = {}
            print("New index created.")

    def normalize_embeddings(self, embeddings):
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings / norms

    def add_embeddings(self, embeddings, metadatas):
        embeddings = self.normalize_embeddings(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        # conncet metadata with IDs
        current_size = len(self.id_to_metadata)
        for i, meta in enumerate(metadatas):
            self.id_to_metadata[current_size + i] = meta
            
        # persistently save to disk
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.id_to_metadata, f)
        print(f"{len(metadatas)} Chunks safed to {self.persist_directory}")

    # ajusts the embedding shape because FAISS is working with 2D arrays --> convertion of 1D query embedding to 2D array with shape (1, embedding_dim)
    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        query_embedding = query_embedding.reshape(1, -1)
        query_embedding = self.normalize_embeddings(query_embedding)
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        results = [(self.id_to_metadata[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
        return results