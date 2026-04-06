"""
ReefGuide RAG Pipeline.

Orchestrates the question-answering pipeline:
1. Greeting detection - returns response without retrieval 
2. GBR topic classification - filters off-topic questions
3. Document retrieval - searches vector database
4. Answer generation - produces answer with sources

Uses Ollama LLM and FAISS for vector search.
"""

from typing import List, Dict, Any, Optional
from src.embedding_manager import EmbeddingManager
from src.faiss_vector_store import FaissVectorStore
from src.retriever import RAGRetriever
from src.utils import load_config
from src.utils import setup_logger
import re

config = load_config()

logger = setup_logger(__name__)

# ============================================================================
# LLM integration
# ============================================================================

from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
load_dotenv()

llm = OllamaLLM(
    model=config['llm']['model'],
    temperature=config['llm']['temperature'],
    top_p=config['llm']['top_p']
)

# ============================================================================
# Detection functions
# ============================================================================

def is_greeting(query: str, keywords: List[str]) -> bool:
    """
    Check if user input is a casual greeting using keyword list.
    
    Args:
        query: User input text to classify
        keywords: List of greeting keywords/phrases from config
    
    Returns:
        True if input matches greeting keywords, otherwise False
    """
    clean_query = re.sub(r'[^\w\s]', '', query.lower())
    words = clean_query.split()
    
    if not words:
        return False
        
    if words[0] in keywords:
        return True

    if any(phrase in clean_query for phrase in keywords if " " in phrase):
        return True
        
    return False


def classify_gbr_question(query: str) -> bool:
    """
    Classifies if user question is about GBR topics.
    Off-topic questions do not trigger retrieval.
    
    Args:
        query: User question text
    
    Returns:
        True if question is GBR-related, False if off-topic
    """
    classification_prompt = f"""Is this user question asking for information about the Great Barrier Reef, GBR, marine life, ocean ecosystems, conservation, tourism, employment, infrastructure, fish, coral, or related topics?
Answer with only: YES or NO

Question: {query}
Answer:"""
        
    try:
        classification = llm.invoke(classification_prompt).strip().lower()
        return "yes" in classification
    except Exception as e:
        logger.warning(f"Classification failed, defaulting to True: {e}")
        return True 

# ============================================================================
# RAG pipeline functions
# ============================================================================

def retrieval_query(query: str, retriever: RAGRetriever, top_k: Optional[int] = None, score_threshold: Optional[float] = None, return_context: bool = False) -> Dict[str, Any]:
    """
    RAG Pipeline: Search vector database and generate answer.
    Used by get_answer() after query classification.
    
    Main functions:
    1. Searches in vector DB for relevant documents
    2. Checks confidence score against threshold
    3. Generates and returns answer with sources 
    
    Args:
        query: User question
        retriever: RAGRetriever instance
        top_k: Number of top documents to retrieve (uses config default if None)
        score_threshold: Minimum similarity score (uses config default if None)
        return_context: Whether to include raw context in response
    
    Returns:
        Dict containing:
            - response: Generated answer text
            - sources: List of source documents with metadata
            - confidence: Highest similarity score
            - context: Raw context (if return_context=True)
    """

    if top_k is None:
        top_k = config['retrieval']['top_k']
    if score_threshold is None:
        score_threshold = config['retrieval']['score_threshold']

    results = retriever.retrieve(query=query, top_k=top_k, score_threshold=score_threshold)
    
    if not results:
        return {
            'response': "No relevant documents found to answer the question.",
            'sources': [],
            'confidence': 0.0
        }

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
    
    if confidence < score_threshold:
        return {
            'response': "Unfortunately the documents I found aren't relevant enough to provide an accurate answer.",
            'sources': sources,
            'confidence': round(float(confidence), 3)
        }
    
    context = "\n\n".join([doc['content'] for doc in results])
      
    prompt = f"""Use the following context to answer the question concisely and factually. 
Do not say where the information comes from, just give the answer. 
If the provided texts mention different numbers or information for the same topic, list them separately.
Only mention missing information if the user specifically asks about it. Do not add disclaimers about what you don't know.
Keep the answer to 1–3 sentences.

Context: {context}

Question: {query}

Answer:"""
    
    try:
        response = llm.invoke(prompt)
    except Exception as e:
        logger.error(f"LLM invocation failed: {e}")
        return {
            'response': f"Error generating answer: {str(e)}",
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

# ============================================================================
# Orchestration function
# ============================================================================

_embedding_manager = None
_vector_store = None
_retriever = None

def get_answer(query: str, top_k: Optional[int] = None, score_threshold: Optional[float] = None) -> Dict[str, Any]:
    """
    Returns user's answer.
    
    Main orchestration function that:
    1. Detects greetings
    2. Classifies if question is about GBR 
    3. Retrieves relevant documents (vector search)
    4. Generates and returns answer with metadata

    
    Args:
        query: User input text or question
        top_k: Number of top documents to retrieve (uses config default if None)
        score_threshold: Minimum similarity score (uses config default if None)
    
    Returns:
        Dict with keys:
            - response: Generated answer text
            - sources: List of source documents (empty for non-RAG queries)
            - confidence: Similarity score
            - is_greeting: True if input was a greeting
            - skip_sources: True if sources should not be displayed
    
    """
    global _embedding_manager, _vector_store, _retriever
   
    logger.info(f"Query received | len={len(query)}")
    
    greeting_keywords = list(config.get('greeting', {}).get('keywords', []))
    logger.debug(f"Loaded {len(greeting_keywords)} greeting keywords: {greeting_keywords[:5]}...")
    
    is_greeting_result = is_greeting(query, greeting_keywords)
    logger.info(f"is_greeting('{query}') = {is_greeting_result}")
    
    if is_greeting_result:
        logger.info("Greeting detected - skipping retrieval")
        return {
            'response': "Hi there! Feel free to ask me anything about the Great Barrier Reef!",
            'sources': [],
            'confidence': 0.0,
            'is_greeting': True,
            'skip_sources': True
        }
    
    if not classify_gbr_question(query):
        logger.info("Question classified as off-topic")
        return {
            'response': "I'm specialized in answering questions about the Great Barrier Reef, marine life, and conservation. Feel free to ask me anything about those topics!",
            'sources': [],
            'confidence': 0.0,
            'is_greeting': False,
            'skip_sources': True
        }
    
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
                'confidence': 0.0,
                'is_greeting': False,
                'skip_sources': True
            }
    
    try:
        retriever = _retriever
        result = retrieval_query(query, retriever, top_k, score_threshold, return_context=True)
        
        result['is_greeting'] = False
   
        threshold = score_threshold if score_threshold is not None else config['retrieval']['score_threshold']
        result['skip_sources'] = result.get('confidence', 0.0) < threshold
        
        logger.info(f"Query completed | confidence={result.get('confidence', 0):.3f}")
        return result
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)[:100]}"
        logger.error(error_msg)
        return {
            'response': f"Error: {error_msg}",
            'sources': [],
            'confidence': 0.0,
            'is_greeting': False,
            'skip_sources': True
        }


