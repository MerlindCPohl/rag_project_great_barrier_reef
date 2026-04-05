# %%
# 1. Setup and Imports

import pymupdf as pdf
import os
import sys
sys.path.insert(0, '../')  

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from src.embedding_manager import EmbeddingManager
from src.utils import extract_text_from_pdf, clean_text_for_bge, remove_duplicate_chunks, get_chunk_hash, load_metadata_from_config, detect_language, load_config, setup_logger
from src.faiss_vector_store import FaissVectorStore

# Initialize logging
logger = setup_logger(__name__)

# Load configuration
config = load_config()


# Paths
pdf_path = "../data/Access Economics 2007 Economic contribution of Great Barrier Reef Marine Park 2005-2006 Kopie.pdf"
document_output_path = "gbr_extracted_text.txt"

# Page selection: pages 6 – 86 
selected_pages = list(range(5, 87))

# Load metadata from config file
filename = os.path.basename(pdf_path)
metadata = load_metadata_from_config(filename)


# %%
# 2. Execute PDF extraction

extract_text_from_pdf(pdf_path, selected_pages, document_output_path)

logger.info(f"PDF extraction complete")

# %%
# 3. Load text file with previously defined metadata

loader = TextLoader(document_output_path, encoding="utf-8")
docs = loader.load()

# Add metadata to documents
for doc in docs:
    doc.metadata.update(metadata)



# %%
# 4. Data cleaning: white space removal
# Boilerplate removal: all emails, urls, page numbers, copyright info, PDF generation info

for doc in docs:
    doc.page_content = clean_text_for_bge(doc.page_content)


test = "Text... © Great Barrier Reef Marine Park Authority Page 5 of 10"
logger.info(clean_text_for_bge(test))


# %%
# 5. Language detection (tags each document with the language code)

language_counts = {}
for doc in docs:
    language = detect_language(doc.page_content)
    doc.metadata['language'] = language
    language_counts[language] = language_counts.get(language, 0) + 1

logger.info(f"Language detection complete. Distribution={language_counts}")


# %%
# 6. Data exploration: analyze text before chunking (how often does one word appear, how long are the documents, any outliers?)

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

# Basic statistics
total_chars = sum(len(doc.page_content) for doc in docs)
total_words = sum(len(doc.page_content.split()) for doc in docs)
doc_lengths = [len(doc.page_content) for doc in docs]

logger.info("=== Data Overview ===")
logger.info(f"Total documents: {len(docs)}")
logger.info(f"Total characters: {total_chars:,}")
logger.info(f"Total words: {total_words:,}")
logger.info(f"Avg doc length: {statistics.mean(doc_lengths):,.0f} chars")
logger.info(f"Min/Max doc length: {min(doc_lengths):,} / {max(doc_lengths):,} chars")

# Extract meaningful keywords (filtered)
logger.info("\n=== Top 10 Keywords (filtered) ===")
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Filter: only alphabetic characters, length > min_word_length, no stopwords
min_word_length = config['data_exploration']['min_word_length']
all_words = [w for w in " ".join(doc.page_content.lower() for doc in docs).split() 
             if w.isalpha() and len(w) > min_word_length]
keywords = [w for w in all_words if w not in stop_words]

keyword_freq = Counter(keywords)
top_keywords_count = config['data_exploration']['top_keywords_count']
for word, count in keyword_freq.most_common(top_keywords_count):
    logger.info(f"  {word}: {count}")

# Check for very short documents (potential outliers)
logger.info("\n=== Potential Issues ===")
short_threshold = config['data_exploration']['short_document_threshold']
short_docs = [d for d in docs if len(d.page_content) < short_threshold]
if short_docs:
    logger.info(f" Found {len(short_docs)} very short documents (< {short_threshold} chars)")
    logger.info(f"  First example: '{short_docs[0].page_content}'")
else:
    logger.info("No unusually short documents")

# %%
# 7. Initialize embedding manager (before semantic chunking)

try:
    embedding_manager = EmbeddingManager()
    logger.info("EmbeddingManager initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize EmbeddingManager: {e}")
    embedding_manager = None

# %%
# 8. Semantic chunking - creates chunks via semantic sense

from langchain_experimental.text_splitter import SemanticChunker

# Use semantic chunker - breaks text into chunks with semnatic coherence 
semantic_splitter = SemanticChunker(
    embedding_manager.model,
    breakpoint_threshold_type=config['ingestion']['breakpoint_threshold_type'],
    breakpoint_threshold_amount=config['ingestion']['breakpoint_threshold_amount']
)

chunks = semantic_splitter.split_documents(docs)

for i, chunk in enumerate(chunks):
    logger.info(f"Chunk {i}:")
    logger.info(f"  Content: {chunk.page_content[:200]}...")  # First 200 chars
    logger.info(f"  Metadata: {chunk.metadata}")
    logger.info()


# %%
# 9. Delete old vector store before re-embedding with semantic chunks

import shutil
import os

vector_store_path = "../data/vector_store"
if os.path.exists(vector_store_path):
    shutil.rmtree(vector_store_path)
    logger.info(f"Deleted old vector store at {vector_store_path}")
else:
    logger.debug("No existing vector store found (will create new one)")


# %%
# 10. Remove duplicates in data if exist

chunks = remove_duplicate_chunks(chunks)

# %%
# 11. Initialize vector store

vector_store = FaissVectorStore(embedding_dim=embedding_manager.get_embedding_dimension())
logger.info("Vector store initialized successfully")

# %%
# 12. Check for duplicate chunks before embedding

# Get hashes of new chunks
new_chunk_hashes = {get_chunk_hash(chunk.page_content): chunk for chunk in chunks}

# Get existing chunk hashes from vector store (what is already stored)
existing_hashes = set()
for metadata in vector_store.id_to_metadata.values():
    if "chunk_hash" in metadata:
        existing_hashes.add(metadata["chunk_hash"])

# Filter: only new chunks will be embedded to avoid double embeddings
chunks_to_add = [chunk for chunk_hash, chunk in new_chunk_hashes.items() 
                 if chunk_hash not in existing_hashes]

logger.info(f"Total chunks: {len(chunks)}")
logger.info(f"Already in vector store: {len(chunks) - len(chunks_to_add)}")
logger.info(f"New chunks to add: {len(chunks_to_add)}")


# %%
#13. embed remaining chunks after removal of duplicates into created vector store and save to disk

if len(chunks_to_add) == 0:
    logger.debug("No new chunks to add (all chunks already in vector store)")
else:
    chunk_texts = [chunk.page_content for chunk in chunks_to_add]
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
            "chunk_hash": get_chunk_hash(chunk.page_content)
        } 
        for chunk in chunks_to_add
    ]
    vector_store.add_embeddings(embeddings, metadatas)

    logger.info(f"Created {len(chunks_to_add)} chunks from {len(docs)} documents")
    logger.info(f"Chunk sizes: min={min(len(c.page_content) for c in chunks_to_add)}, max={max(len(c.page_content) for c in chunks_to_add)}")


