"""
Retrieves relevant documents from FAISS vector store for a given query.
Uses semantic similarity search to find top-k documents matching the user's query.
Filters results by similarity threshold before returning.
"""

from typing import List, Dict, Any
from src.utils import setup_logger
from src.faiss_vector_store import FaissVectorStore
from src.embedding_manager import EmbeddingManager

logger = setup_logger(__name__) 


class RAGRetriever:
    
    def __init__(self, vector_store: FaissVectorStore, embedding_manager: EmbeddingManager):
        
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int, score_threshold: float) -> List[Dict[str, Any]]:
       
        logger.debug(f"Retrieving documents for query: '{query}' with top_k={top_k} and score_threshold={score_threshold}")
        
        try:
            query_embeddings = self.embedding_manager.generate_embeddings([query])
            query_embedding = query_embeddings[0]
            
            results = self.vector_store.search(query_embedding, top_k=top_k)
    
            retrieved_docs = []
            for rank, (metadata, similarity_score) in enumerate(results, 1):
                if similarity_score >= score_threshold:
                    retrieved_docs.append({
                        'id': rank,
                        'content': metadata.get('content', ''),
                        'source': metadata.get('source', ''),
                        'similarity_score': float(similarity_score),
                        'metadata': metadata,
                        'rank': rank
                    })
            
            if retrieved_docs:
                logger.info(f"Retrieved {len(retrieved_docs)} documents above similarity threshold {score_threshold}")
            else:
                logger.debug(f"No documents above similarity threshold {score_threshold}")
            
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Retrieval error: {str(e)[:100]}")
            return []