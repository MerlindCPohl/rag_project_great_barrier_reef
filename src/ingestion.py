"""Data ingestion pipeline. 

Handling of:
1. PDF extraction and text loading
2. Text cleaning and preprocessing
3. Language detection
4. Data exploration and validation
5. Semantic chunking
6. Embedding generation
7. Vector store persistence
"""

import os
import sys
import shutil
sys.path.insert(0, '../')  
from langchain_community.document_loaders import TextLoader
from src.embedding_manager import EmbeddingManager
from src.utils import extract_text_from_pdf, clean_text_for_bge, remove_duplicate_chunks,load_metadata_from_config, detect_language, load_config, setup_logger
from src.faiss_vector_store import FaissVectorStore

logger = setup_logger(__name__)

config = load_config()

# ============================================================================
# Configure paths and metadata
# ============================================================================

pdf_path = "../data/Measuring_the_economic_financial_value_of_the_Great_Barrier_Reef_Marine_Park_2005-06.pdf"
document_output_path = "gbr_extracted_text.txt"

selected_pages = list(range(5, 87))

filename = os.path.basename(pdf_path)
metadata = load_metadata_from_config(filename)

# ============================================================================
# Extract text from PDF
# ============================================================================

extract_text_from_pdf(pdf_path, selected_pages, document_output_path)
logger.info(f"PDF extraction complete")

# ============================================================================
# Load extracted text and attach metadata
# ============================================================================

loader = TextLoader(document_output_path, encoding="utf-8")
docs = loader.load()

for doc in docs:
    doc.metadata.update(metadata)

# ============================================================================
# Clean text (remove boilerplate, whitespace, URLs, emails, page numbers)
# ============================================================================

for doc in docs:
    doc.page_content = clean_text_for_bge(doc.page_content)

test = "Text... © Great Barrier Reef Marine Park Authority Page 5 of 10"
logger.info(clean_text_for_bge(test))

# ============================================================================
# Detect language for each document
# ============================================================================

language_counts = {}
for doc in docs:
    language = detect_language(doc.page_content)
    doc.metadata['language'] = language
    language_counts[language] = language_counts.get(language, 0) + 1

logger.info(f"Language detection complete. Distribution={language_counts}")

# ============================================================================
# Data exploration: Analyze dataset statistics, word frequency, and identify potential issues
# ============================================================================

from collections import Counter
import statistics
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

total_chars = sum(len(doc.page_content) for doc in docs)
total_words = sum(len(doc.page_content.split()) for doc in docs)
doc_lengths = [len(doc.page_content) for doc in docs]

logger.info("=== Data Overview ===")
logger.info(f"Total documents: {len(docs)}")
logger.info(f"Total characters: {total_chars:,}")
logger.info(f"Total words: {total_words:,}")
logger.info(f"Avg doc length: {statistics.mean(doc_lengths):,.0f} chars")
logger.info(f"Min/Max doc length: {min(doc_lengths):,} / {max(doc_lengths):,} chars")

logger.info("\n=== Top 10 Keywords (filtered) ===")
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

min_word_length = config['data_exploration']['min_word_length']
all_words = [w for w in " ".join(doc.page_content.lower() for doc in docs).split() 
             if w.isalpha() and len(w) > min_word_length]
keywords = [w for w in all_words if w not in stop_words]

keyword_freq = Counter(keywords)
top_keywords_count = config['data_exploration']['top_keywords_count']
for word, count in keyword_freq.most_common(top_keywords_count):
    logger.info(f"  {word}: {count}")

logger.info("\n=== Potential Issues ===")
short_threshold = config['data_exploration']['short_document_threshold']
short_docs = [d for d in docs if len(d.page_content) < short_threshold]
if short_docs:
    logger.info(f" Found {len(short_docs)} very short documents (< {short_threshold} chars)")
    logger.info(f"  First example: '{short_docs[0].page_content}'")
else:
    logger.info("No unusually short documents")

# ============================================================================
# Load SentenceTransformer model for generating embeddings
# ============================================================================

try:
    embedding_manager = EmbeddingManager()
    logger.info("EmbeddingManager initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize EmbeddingManager: {e}")
    embedding_manager = None

# ============================================================================
# Use semantic chunking to preserve meaning
# ============================================================================
# 
from langchain_experimental.text_splitter import SemanticChunker

semantic_splitter = SemanticChunker(
    embedding_manager.model,
    breakpoint_threshold_type=config['ingestion']['breakpoint_threshold_type'],
    breakpoint_threshold_amount=config['ingestion']['breakpoint_threshold_amount']
)

chunks = semantic_splitter.split_documents(docs)
logger.info(f"Created {len(chunks)} chunks via semantic chunking")

# ============================================================================
# Clean up old vector store before re-embedding
# ============================================================================

vector_store_path = "../data/vector_store"
if os.path.exists(vector_store_path):
    shutil.rmtree(vector_store_path)
    logger.info(f"Deleted old vector store at {vector_store_path}")
else:
    logger.debug("No existing vector store found to delete")

# ============================================================================
# Remove duplicate chunks within current dataset
# ============================================================================

chunks = remove_duplicate_chunks(chunks)

# ============================================================================
# Initialize FAISS vector store
# ============================================================================

vector_store = FaissVectorStore(embedding_dim=embedding_manager.get_embedding_dimension())
logger.info("Vector store initialized successfully")

# ============================================================================
# Generate embeddings and persist to vector store
# ============================================================================

chunk_texts = [chunk.page_content for chunk in chunks]
embeddings = embedding_manager.generate_embeddings(chunk_texts) 
    
metadatas = [
    {
        "content": chunk.page_content,  
        "source": chunk.metadata.get("source", ""),  
        "title": chunk.metadata.get("title", ""), 
        "author": chunk.metadata.get("author", ""),  
        "year": chunk.metadata.get("year", ""), 
        "description": chunk.metadata.get("description", ""),  
        "categories": chunk.metadata.get("categories", []),  
        "language": chunk.metadata.get("language", "en"), 
    } 
    for chunk in chunks
]
    
vector_store.add_embeddings(embeddings, metadatas)

logger.info(f"Successfully created index with {len(chunks)} chunks.")

