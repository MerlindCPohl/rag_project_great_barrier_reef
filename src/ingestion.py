# %%
# 1. Setup and Imports

import pymupdf as pdf
import re
import os
import glob
import sys
sys.path.insert(0, '../')  

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from typing import Optional, Dict, Any
from src.embedding_manager import EmbeddingManager
from src.utils import extract_text_from_pdf, clean_text_for_bge, remove_duplicate_chunks, get_chunk_hash, load_metadata_from_config, detect_language, load_config
from src.vector_store import FaissVectorStore
from src.logging_config import setup_logger

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

print(f"PDF extraction complete")

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
print("\n=== Top 10 Keywords (filtered) ===")
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
    print(f"  {word}: {count}")

# Check for very short documents (potential outliers)
print("\n=== Potential Issues ===")
short_threshold = config['data_exploration']['short_document_threshold']
short_docs = [d for d in docs if len(d.page_content) < short_threshold]
if short_docs:
    print(f" Found {len(short_docs)} very short documents (< {short_threshold} chars)")
    print(f"  First example: '{short_docs[0].page_content}'")
else:
    print("No unusually short documents")

# %%
# 7. Semantic chunking - creates chunks via semantic sense

from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize embeddings for semantic chunking
embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-m3")

# Use semantic chunker - breaks at natural semantic boundaries instead of fixed size
semantic_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type=config['ingestion']['breakpoint_threshold_type'],
    breakpoint_threshold_amount=config['ingestion']['breakpoint_threshold_amount']
)

chunks = semantic_splitter.split_documents(docs)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i}:")
    print(f"  Content: {chunk.page_content[:200]}...")  # First 200 chars
    print(f"  Metadata: {chunk.metadata}")
    print()


# %%
# 8. Delete old vector store before re-embedding with semantic chunks

import shutil
import os

vector_store_path = "../data/vector_store"
if os.path.exists(vector_store_path):
    shutil.rmtree(vector_store_path)
    logger.info(f"Deleted old vector store at {vector_store_path}")
else:
    logger.debug("No existing vector store found (will create new one)")


# %%
# 9. Remove duplicates in data if exist

chunks = remove_duplicate_chunks(chunks)

# %%
# 10. Initialize embeddings and create vector store

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# %%
# 11. Initialize embedding manager

try:
    embedding_manager = EmbeddingManager()
    print("EmbeddingManager initialized successfully")
except Exception as e:
    print(f"Failed to initialize EmbeddingManager: {e}")
    embedding_manager = None


# %%
# 12. Initialize vector store

vector_store = FaissVectorStore(embedding_dim=embedding_manager.get_embedding_dimension())
print("Vector store initialized successfully")

# %%
# 13. Check for duplicate chunks before embedding

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
#14. embed remaining chunks after removal of duplicates into created vector store and save to disk

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

# %%
#15. Repeat steps 1 - 12 automaticallyin case that new documents are added 


def process_new_documents(pdf_directory: Optional[str] = None, vector_store: Optional[FaissVectorStore] = None, embedding_manager: Optional[EmbeddingManager] = None) -> Dict[str, Any]:
    
    if pdf_directory is None:
        # Use the data directory relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        pdf_directory = os.path.join(project_root, "data")
    
    if vector_store is None or embedding_manager is None:
        logger.error("Error: vector_store and embedding_manager must be provided")
        return {"status": "failed", "message": "Missing vector_store or embedding_manager"}
    
    # Find all PDF files
    pdf_files = glob.glob(os.path.join(pdf_directory, "*.pdf"))
    
    if not pdf_files:
        return {"status": "skipped", "message": "No PDF files found", "files_processed": 0}
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    # Get existing file hashes from metadata
    existing_sources = set()
    for metadata in vector_store.id_to_metadata.values():
        if "source" in metadata:
            existing_sources.add(metadata["source"])
    
    # Filter new PDFs
    new_pdfs = [pdf for pdf in pdf_files if os.path.basename(pdf) not in existing_sources]
    
    if not new_pdfs:
        return {"status": "skipped", "message": "All PDFs already in vector store", "files_processed": 0}
    
    logger.info(f"Processing {len(new_pdfs)} new PDF(s)...")
    
    total_chunks_added = 0
    
    # Process each new PDF
    for pdf_path in new_pdfs:
        logger.info(f"\nProcessing: {os.path.basename(pdf_path)}")
        
        try:
            # Extract text
            selected_pages = list(range(0, 100))  # Adjust as needed
            temp_output = f"{os.path.basename(pdf_path)}.txt"
            extract_text_from_pdf(pdf_path, selected_pages, temp_output)
            
            # Load text file
            loader = TextLoader(temp_output, encoding="utf-8")
            new_docs = loader.load()
            
            # Clean text content
            for doc in new_docs:
                doc.page_content = clean_text_for_bge(doc.page_content)
            
            # Detect language and add to metadata
            for doc in new_docs:
                language = detect_language(doc.page_content)
                doc.metadata['language'] = language
            
            # Load metadata from config and apply to all docs
            document_metadata = load_metadata_from_config(os.path.basename(pdf_path))
            for doc in new_docs:
                doc.metadata.update(document_metadata)
            
            # Split into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
            new_chunks = splitter.split_documents(new_docs)
            new_chunks = remove_duplicate_chunks(new_chunks)
            
            # Check for duplicates with existing chunks
            new_chunk_hashes = {get_chunk_hash(chunk.page_content): chunk for chunk in new_chunks}
            existing_hashes = set()
            for metadata in vector_store.id_to_metadata.values():
                if "chunk_hash" in metadata:
                    existing_hashes.add(metadata["chunk_hash"])
            
            chunks_to_add = [chunk for chunk_hash, chunk in new_chunk_hashes.items()
                           if chunk_hash not in existing_hashes]
            
            if chunks_to_add:
                # Embed and add to vector store
                chunk_texts = [chunk.page_content for chunk in chunks_to_add]
                embeddings = embedding_manager.generate_embeddings(chunk_texts)
                
                # Load metadata from config for consistent source info
                document_metadata = load_metadata_from_config(os.path.basename(pdf_path))
                
                metadatas = [
                    {
                        "content": chunk.page_content,
                        "source": document_metadata.get("source", os.path.basename(pdf_path)),
                        "title": document_metadata.get("title", ""),
                        "author": document_metadata.get("author", ""),
                        "year": document_metadata.get("year", ""),
                        "description": document_metadata.get("description", ""),
                        "categories": document_metadata.get("categories", []),
                        "language": chunk.metadata.get("language", "en"),
                        "chunk_hash": get_chunk_hash(chunk.page_content)
                    }
                    for chunk in chunks_to_add
                ]
                vector_store.add_embeddings(embeddings, metadatas)
                total_chunks_added += len(chunks_to_add)
                logger.info(f"Added {len(chunks_to_add)} chunks from {os.path.basename(pdf_path)}")
            else:
                logger.info(f"No new chunks to add from {os.path.basename(pdf_path)}")
            
            # Clean up temp file
            if os.path.exists(temp_output):
                os.remove(temp_output)
                
        except Exception as e:
            logger.error(f"Error processing {os.path.basename(pdf_path)}: {e}")
            continue
    
    return {
        "status": "success",
        "files_processed": len(new_pdfs),
        "total_chunks_added": total_chunks_added,
        "message": f"Processed {len(new_pdfs)} PDF(s) and added {total_chunks_added} chunks"
    }


