# %%
#1. Retrieval: gives results from vector store based on query
# Retrieve relevant documents from the vector store in form of a list of dictionaries 
# query is tranformed into an embedding and then used to search the vector store for similar documents

import sys
sys.path.insert(0, '../')

from typing import List, Dict, Any, Optional
from src.embedding_manager import EmbeddingManager
from src.vector_store import FaissVectorStore
from src.utils import load_config
from src.logging_config import setup_logger
import time

# Load configuration
config = load_config()

# Initialize logging
logger = setup_logger(__name__)

class RAGRetriever:
    def __init__(self, vector_store: FaissVectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
       
        start_time = time.time()
        logger.debug(f"Retrieving documents for query: '{query}' with top_k={top_k} and score_threshold={score_threshold}")
        
        try:
            # Generate embedding for the query with retry logic
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
            
            for rank, (metadata, distance) in enumerate(results, 1):
                similarity_score = distance
                
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
                logger.info(f"Retrieved {len(retrieved_docs)} documents above threshold of {score_threshold}")
            else:
                logger.debug(f"No documents above threshold {score_threshold}")
            
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Retrieval error: {str(e)[:100]}")
            return []

# %%
#2. Initialize vector store and embedding manager (load from disk)

try:
    embedding_manager = EmbeddingManager()
    logger.info("EmbeddingManager initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize EmbeddingManager: {e}")
    embedding_manager = None

try:
    vector_store = FaissVectorStore(embedding_dim=embedding_manager.get_embedding_dimension())
    logger.info(f"Vector store loaded with {len(vector_store.id_to_metadata)} chunks")
except Exception as e:
    logger.error(f"Failed to initialize Vector Store: {e}")
    vector_store = None

# %%
# 3. Initialize RAGRetriever with instances from ingestion.ipynb

rag_retriever = RAGRetriever(vector_store=vector_store, embedding_manager=embedding_manager)

query = "What is the economic contribution of the Great Barrier Reef?"
results = rag_retriever.retrieve(query=query, top_k=5, score_threshold=0.0)

for doc in results:
    
    logger.info(f"\nRank {doc['rank']}: {doc['similarity_score']:.4f}")
    logger.info(f"Source: {doc['source']}")
    logger.info(f"Content preview: {doc['content'][:200]}...")


# %%
#4. LLM integration with Ollama

from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
load_dotenv()

# Initialize LLM with config values
llm = OllamaLLM(
    model=config['llm']['model'],
    temperature=config['llm']['temperature'],
    top_p=config['llm']['top_p']
)


# %%
#5. RAG function for information retrieval with minimal instructions


def invoke_llm_with_retry(llm: OllamaLLM, prompt: str, max_retries: int = 2, timeout: int = 60) -> str:
   
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Invoking LLM (attempt {attempt + 1}/{max_retries + 1})...")
            start_time = time.time()
            
            response = llm.invoke(prompt)
            
            elapsed = time.time() - start_time
            logger.info(f"LLM response received in {elapsed:.1f}s")
            return response
            
        except TimeoutError as e:
            if attempt < max_retries:
                wait_time = 2 ** attempt
                logger.debug(f"LLM timeout (attempt {attempt + 1}): exceeded {timeout}s")
                logger.debug(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"LLM timeout after {max_retries + 1} attempts")
                raise RuntimeError(f"LLM call timed out after {max_retries + 1} attempts")
                
        except Exception as e:
            if attempt < max_retries:
                wait_time = 2 ** attempt
                logger.debug(f"LLM error (attempt {attempt + 1}/{max_retries + 1}): {str(e)[:100]}")
                logger.debug(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"LLM invocation failed after {max_retries + 1} attempts: {str(e)}")
                raise RuntimeError(f"LLM invocation failed after {max_retries + 1} attempts: {str(e)}")


def retrieval_query(query: str, retriever: RAGRetriever, llm: OllamaLLM, top_k: Optional[int] = None, score_threshold: Optional[float] = None, return_context: bool = False) -> Dict[str, Any] | str:

    # Use config defaults if not provided
    if top_k is None:
        top_k = config['retrieval']['top_k']
    if score_threshold is None:
        score_threshold = config['retrieval']['score_threshold']

    results = retriever.retrieve(query=query, top_k=top_k, score_threshold=score_threshold)
    
    if not results:
        return "No relevant documents found to answer the question."

    confidence = max(doc['similarity_score'] for doc in results)

    sources = []
    for doc in results:
        clean_preview = doc['content'][:150].replace('\n', ' ').strip()
        sources.append({
            'title': doc['metadata'].get('title', 'Unknown'),
            'source': doc['source'],
            'language': doc['metadata'].get('language', 'en'),
            'score': round(float(doc['similarity_score']), 3),
            'preview': f"{clean_preview}..."
        })
    
    
    # in case of low confidence,choose to return the sources with a note about low confidence instead of an answer to avoid errors
    if confidence < score_threshold:
        return {
            'answer': "The found documents are not sufficiently relevant. I refuse to answer to avoid errors.",
            'sources': sources,
            'confidence': round(float(confidence), 3)
        }
    
    context = "\n\n".join([doc['content'] for doc in results])
    
    # generate answer    
    prompt = f"""
    Use the following context to answer the question concisely and factually. 
    Do not say where the information comes from, just give the answer. 
    If the provided texts mention different numbers or information for the same topic, list them separately. 
    If the context does not contain the answer, say: "I don't know based on the provided context."
    Keep the answer to 1–3 sentences.

        Context: {context}

        Question: {query}

        Answer:"""
    
    try:
        response = invoke_llm_with_retry(llm, prompt, max_retries=2)
    except RuntimeError as e:
        logger.error(f"LLM invocation failed: {e}")
        return {
            'answer': f"Error generating answer: {str(e)}",
            'sources': sources,
            'confidence': round(float(confidence), 3)
        }

    output = {
        'response': response,
        'sources': sources,
        'confidence': round(float(confidence), 3)
    }
    
    if return_context:
        output['context'] = context
    
    return output
  

# %%
#6. Output query with sources and confidence score
# Note: This test code is commented out to prevent execution on module import

# result = retrieval_query("How much will the Earth still warm up?", rag_retriever, llm, return_context=True)

# print("\n" + "_"*100)
# print("")
# print("Answer:")
# print("")
# print(result['response'] if isinstance(result, dict) else result)

# print("\n" + "_"*100)
# print("")
# print("Sources:")
# print("")
# if isinstance(result, dict):
#     for i, source in enumerate(result['sources'], 1):
#         print(f"\n[Source {i}]")
#         print(f"File: {source['source']}")
#         print(f"Score: {source['score']}")
#         print(f"Preview:{source['preview']}")

# print("\n" + "_"*100)
# if isinstance(result, dict):
#     print(f"Confidence Score: {result['confidence']}")
# print("_"*100)


# %%
# 7. Answering user queries via Streamlit 

def get_answer(query: str, top_k: Optional[int] = None, score_threshold: Optional[float] = None) -> Dict[str, Any]:
    """
    Main entry point for answering user queries.
    
    Handles initialization, error recovery, and graceful degradation.
    
    Args:
        query: User question
        top_k: Number of documents to retrieve (uses config default if None)
        score_threshold: Similarity threshold (uses config default if None)
        
    Returns:
        Dictionary with 'response', 'sources', and 'confidence' keys
    """
    # Use config defaults if not provided
    if top_k is None:
        top_k = config['retrieval']['top_k']
    if score_threshold is None:
        score_threshold = config['retrieval']['score_threshold']
    
    logger.info(f"Query received | len={len(query)}")
    
    try:
        # Initialize components with error handling
        try:
            embedding_manager = EmbeddingManager()
        except Exception as e:
            logger.critical(f"Failed to initialize embedding manager: {e}")
            return {
                'response': "Error: Could not initialize embedding system",
                'sources': [],
                'confidence': 0.0
            }
        
        try:
            vector_store = FaissVectorStore(embedding_dim=embedding_manager.get_embedding_dimension())
        except Exception as e:
            logger.critical(f"Failed to initialize vector store: {e}")
            return {
                'response': "Error: Could not load vector database",
                'sources': [],
                'confidence': 0.0
            }
        
        retriever = RAGRetriever(vector_store, embedding_manager)
        
        # Use retrieval function with error handling
        result = retrieval_query(query, retriever, llm, top_k, score_threshold, return_context=True)
        
        # Normalize result
        if isinstance(result, str):
            logger.warning(f"Retrieval returned: {result[:50]}...")
            return {
                'response': result,
                'sources': [],
                'confidence': 0.0
            }
        
        logger.info(f"Query completed | confidence={result.get('confidence', 0):.3f}")
        return result
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)[:100]}"
        logger.error(error_msg)
        return {
            'response': f"Error: {error_msg}",
            'sources': [],
            'confidence': 0.0
        }


