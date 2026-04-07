"""
ReefGuide RAG Pipeline: Main orchestration and question-answering logic.

Handling of:
1. is_greeting(): Detects casual greetings using keyword matching
2. classify_gbr_question(): LLM-based topic classification (GBR vs off-topic)
3. format_sources(): Formats retrieval results for UI display
4. init_components(): Lazy initialization of RAG components
5. retrieval_query(): Core RAG pipeline (retrieve → format → generate answer)
6. get_answer(): Main entry point and orchestrator

Uses Ollama LLM and FAISS (via RAGRetriever) for vector search.
All user-facing messages externalized to config.yaml
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
    classification_prompt = f"""You are ReefGuide, deployed at the Great Barrier Reef Marine Park visitor center in Queensland, Australia.
When someone says "here" or "local" or "this place", they refer to the Great Barrier Reef.
Is this user question asking for information about the Great Barrier Reef, GBR, marine life, ocean ecosystems, conservation, tourism, employment, infrastructure, fish, coral, or related topics?

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
# Helper functions
# ============================================================================

def format_sources(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format retrieval results into source metadata."""
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
    return sources


def init_components() -> None:
    """Initialize RAG components (lazy loading)."""
    global _embedding_manager, _vector_store, _retriever
    
    try:
        _embedding_manager = EmbeddingManager()
        _vector_store = FaissVectorStore(embedding_dim=_embedding_manager.get_embedding_dimension())
        _retriever = RAGRetriever(_vector_store, _embedding_manager)
        logger.info("RAG components initialized successfully")
    except Exception as e:
        logger.critical(f"Failed to initialize RAG components: {e}")
        raise



# ============================================================================
# RAG Pipeline
# ============================================================================

def retrieval_query(query: str, retriever: RAGRetriever, top_k: int, 
                   score_threshold: float) -> Dict[str, Any]:
    """
    Core RAG pipeline: retrieve documents, format sources, generate answer. 
    
    Args:
        query: User question
        retriever: RAGRetriever instance
        top_k: Number of documents to retrieve
        score_threshold: Minimum similarity score
    
    Returns:
        Dict with response, sources, confidence, skip_sources
    """

    results = retriever.retrieve(query=query, top_k=top_k, score_threshold=score_threshold)
    
    if not results:
        return {
            'response': config['messages']['no_documents_found'],
            'sources': [],
            'confidence': 0.0,
            'skip_sources': True
        }

    confidence = max(doc['similarity_score'] for doc in results)
    sources = format_sources(results)
    

    if confidence < score_threshold:
        return {
            'response': config['messages']['low_confidence'],
            'sources': sources,
            'confidence': round(float(confidence), 3),
            'skip_sources': True
        }
    
    context = "\n\n".join([doc['content'] for doc in results])
    
    prompt = f"""You are ReefGuide, a chat assistant at the Great Barrier Reef Marine Park visitor center in Queensland, Australia.
When someone says "here", they refer to the Great Barrier Reef region. 
Use the following context to answer the question concisely and factually. 
Do not say where the information comes from, just give the answer. 
If the provided texts mention different numbers or information for the same topic, list them separately.
Only mention missing information if the user specifically asks about it. Do not add disclaimers about what you don't know.
Keep the answer to 1–3 sentences.

Context: {context}

Question: {query}

Answer:"""
    
    try:
        response = llm.invoke(prompt)
        return {
            'response': response,
            'sources': sources,
            'confidence': round(float(confidence), 3),
            'skip_sources': False
        }
    except Exception as e:
        logger.error(f"LLM invocation failed: {e}")
        return {
            'response': config['messages']['error_generating_answer'],
            'sources': sources,
            'confidence': round(float(confidence), 3),
            'skip_sources': True
        }

# ============================================================================
# Orchestration function
# ============================================================================

_embedding_manager = None
_vector_store = None
_retriever = None

def get_answer(query: str, top_k: Optional[int] = None, 
               score_threshold: Optional[float] = None) -> Dict[str, Any]:
    """
    Main entry point: orchestrates greeting detection → classification → RAG → response.
    
    Args:
        query: User input
        top_k: Number of documents to retrieve (uses config default if None)
        score_threshold: Minimum similarity score (uses config default if None)
    
    Returns:
        Dict with response, sources, confidence, skip_sources, is_greeting
    """
    global _retriever
    
    logger.info(f"Query received | len={len(query)}")
    
    if _retriever is None:
        init_components()
    
    top_k = top_k or config['retrieval']['top_k']
    score_threshold = score_threshold or config['retrieval']['score_threshold']
    greeting_keywords = config.get('greeting', {}).get('keywords', [])
    
    if is_greeting(query, greeting_keywords):
        logger.info("Greeting detected - skipping retrieval")
        return {
            'response': config['messages']['greeting_response'],
            'confidence': 0.0,
            'is_greeting': True,
            'skip_sources': True
        }
    
    if not classify_gbr_question(query):
        logger.info("Question classified as off-topic")
        return {
            'response': config['messages']['off_topic'],
            'sources': [],
            'confidence': 0.0,
            'is_greeting': False,
            'skip_sources': True
        }
    try:
        result = retrieval_query(query, _retriever, top_k, score_threshold)
        result['is_greeting'] = False
        logger.info(f"Query completed | confidence={result.get('confidence', 0):.3f}")
        return result
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)[:100]}")
        return {
            'response': config['messages']['system_error'],
            'sources': [],
            'confidence': 0.0,
            'is_greeting': False,
            'skip_sources': True
        }


