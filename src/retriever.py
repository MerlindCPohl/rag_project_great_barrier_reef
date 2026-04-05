from typing import List, Dict, Any
from src.utils import setup_logger
from src.faiss_vector_store import FaissVectorStore
from src.embedding_manager import EmbeddingManager

logger = setup_logger(__name__) 


class RAGRetriever:
    def __init__(self, vector_store: FaissVectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
       
        logger.debug(f"Retrieving documents for query: '{query}' with top_k={top_k} and score_threshold={score_threshold}")
        
        try:
            # Generate embeddings for query with retry logic
            try:
                query_embeddings = self.embedding_manager.generate_embeddings([query], max_retries=2)
                query_embedding = query_embeddings[0]
            except RuntimeError as e:
                logger.error(f"Failed to embed query after retries: {e}")
                return []
            
            # Search the vector store - returns list of tuples (metadata, distance)
            try:
                results = self.vector_store.search(query_embedding, top_k=top_k)
            except Exception as e:
                logger.error(f"Vector store search failed: {e}")
                raise
            
            retrieved_docs = []
            
            for rank, (metadata, similarity_score) in enumerate(results, 1):
                
                # Note: similarity_score is cosine similarity (0-1, higher = more similar)
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