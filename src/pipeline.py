import sys
sys.path.insert(0, '../')

from typing import List, Dict, Any, Optional
from src.embedding_manager import EmbeddingManager
from src.faiss_vector_store import FaissVectorStore
from src.retriever import RAGRetriever
from src.utils import load_config
from src.utils import setup_logger
import time

# Load configuration
config = load_config()

# Initialize logging
logger = setup_logger(__name__)


# %%
#1. LLM integration with Ollama

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
#2. RAG function for information retrieval with minimal instructions


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


def retrieval_query(query: str, retriever: RAGRetriever, top_k: Optional[int] = None, score_threshold: Optional[float] = None, return_context: bool = False) -> Dict[str, Any] | str:

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
# 3. Answering user queries via Streamlit 

_embedding_manager = None
_vector_store = None
_retriever = None

def get_answer(query: str, top_k: Optional[int] = None, score_threshold: Optional[float] = None) -> Dict[str, Any]:
    global _embedding_manager, _vector_store, _retriever
   
    # initialize just once and reuse to save time 
    if _embedding_manager is None:
        try:
            _embedding_manager = EmbeddingManager()
            _vector_store = FaissVectorStore(embedding_dim=_embedding_manager.get_embedding_dimension())
            _retriever = RAGRetriever(_vector_store, _embedding_manager)
        except Exception as e:
            logger.critical(f"Failed to initialize components: {e}")
            return {
                'response': "Error: Could not initialize system",
                'sources': [],
                'confidence': 0.0
            }
    
    # Use config defaults if not provided
    if top_k is None:
        top_k = config['retrieval']['top_k']
    if score_threshold is None:
        score_threshold = config['retrieval']['score_threshold']
    
    logger.info(f"Query received | len={len(query)}")
    
    try:
        retriever = _retriever
        
        # Use retrieval function with error handling
        result = retrieval_query(query, retriever, top_k, score_threshold, return_context=True)
        
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


