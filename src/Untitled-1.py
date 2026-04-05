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
from src.embedding_manager import EmbeddingManager
from src.utils import extract_text_from_pdf, clean_text_for_bge, remove_duplicate_chunks, get_chunk_hash
from src.vector_store import FaissVectorStore


# Paths
pdf_path = "../data/Access Economics 2007 Economic contribution of Great Barrier Reef Marine Park 2005-2006 Kopie.pdf"
document_output_path = "gbr_extracted_text.txt"

# Page selection: pages 6 – 86 
selected_pages = list(range(5, 87))

# Defining Metadata for the document
metadata = {
    "source": "Access Economics 2007 Economic contribution of Great Barrier Reef Marine Park 2005-2006 Kopie.pdf",
    "author": "Great Barrier Reef Marine Park Authority",
    "year": 2007,
    "description": "A comprehensive study on the economic contribution of the Great Barrier Reef Marine Park for the years 2005-2006.",
}


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
# 5. Data cleaning: white space removal
# Boilerplate removal: all emails, urls, page numbers, copyright info, PDF generation info

for doc in docs:
    doc.page_content = clean_text_for_bge(doc.page_content)


test = "Text... © Great Barrier Reef Marine Park Authority Page 5 of 10"
print(clean_text_for_bge(test))



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

print("=== Data Overview ===")
print(f"Total documents: {len(docs)}")
print(f"Total characters: {total_chars:,}")
print(f"Total words: {total_words:,}")
print(f"Avg doc length: {statistics.mean(doc_lengths):,.0f} chars")
print(f"Min/Max doc length: {min(doc_lengths):,} / {max(doc_lengths):,} chars")

# Extract meaningful keywords (filtered)
print("\n=== Top 10 Keywords (filtered) ===")
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Filter: only alphabetic characters, length > 2, and not stopwords
# check later if words > 2 was a good threshold
all_words = [w for w in " ".join(doc.page_content.lower() for doc in docs).split() 
             if w.isalpha() and len(w) > 2]
keywords = [w for w in all_words if w not in stop_words]

keyword_freq = Counter(keywords)
for word, count in keyword_freq.most_common(10):
    print(f"  {word}: {count}")

# Check for very short documents (potential outliers)
print("\n=== Potential Issues ===")
short_docs = [d for d in docs if len(d.page_content) < 100]
if short_docs:
    print(f" Found {len(short_docs)} very short documents (< 100 chars)")
    print(f" First example: '{short_docs[0].page_content}'")
else:
    print("No unusually short documents")

# %%
# 7. Parsing and chunking

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

chunks = splitter.split_documents(docs)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i}:")
    print(f"  Content: {chunk.page_content[:200]}...")  # First 200 chars
    print(f"  Metadata: {chunk.metadata}")
    print()



# %%
# 7.1. Remove duplicates in data if exist

chunks = remove_duplicate_chunks(chunks)

# %%
# 8. Initialize embeddings and create vector store

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# %%
# 9. Initialize embedding manager

try:
    embedding_manager = EmbeddingManager()
    print("EmbeddingManager initialized successfully")
except Exception as e:
    print(f"Failed to initialize EmbeddingManager: {e}")
    embedding_manager = None


# %%
# 10. Initialize vector store

vector_store = FaissVectorStore(embedding_dim=embedding_manager.get_embedding_dimension())
print("Vector store initialized successfully")

# %%
# 11. Check for duplicate chunks before embedding

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

print(f"Total chunks: {len(chunks)}")
print(f"Already in vector store: {len(chunks) - len(chunks_to_add)}")
print(f"New chunks to add: {len(chunks_to_add)}")


# %%
#12. embed remaining chunks after removal of duplicates into created vecrotr store and save to disk

# Check if all dependencies are available
required_vars = ['chunks_to_add', 'embedding_manager', 'vector_store', 'docs']
missing = [var for var in required_vars if var not in locals()]
if missing:
    print(f"ERROR: Missing variables: {missing}")
    print("Please run cells 1-11 first in order!")
else:
    chunk_texts = [chunk.page_content for chunk in chunks_to_add]
    embeddings = embedding_manager.generate_embeddings(chunk_texts) 
    metadatas = [
        {
            "content": chunk.page_content, 
            "source": chunk.metadata.get("source", ""), 
            "author": chunk.metadata.get("author", ""),
            "year": chunk.metadata.get("year", ""),
            "description": chunk.metadata.get("description", ""),
            "chunk_hash": get_chunk_hash(chunk.page_content)
        } 
        for chunk in chunks_to_add
    ]
    vector_store.add_embeddings(embeddings, metadatas)

    print(f"Created {len(chunks_to_add)} chunks from {len(docs)} documents")
    print(f"Chunk sizes: min={min(len(c.page_content) for c in chunks_to_add)}, max={max(len(c.page_content) for c in chunks_to_add)}")
else:
    print("No new chunks to add (all chunks already in vector store)")

# %%
#13. Repeat steps 1 - 12 automaticallyin case that new documents are added 


def process_new_documents(pdf_directory: str = "../data", vector_store=None, embedding_manager=None):
    
    if vector_store is None or embedding_manager is None:
        print("Error: vector_store and embedding_manager must be provided")
        return {"status": "failed", "message": "Missing vector_store or embedding_manager"}
    
    # Find all PDF files
    pdf_files = glob.glob(os.path.join(pdf_directory, "*.pdf"))
    
    if not pdf_files:
        return {"status": "skipped", "message": "No PDF files found", "files_processed": 0}
    
    print(f"Found {len(pdf_files)} PDF files")
    
    # Get existing file hashes from metadata
    existing_sources = set()
    for metadata in vector_store.id_to_metadata.values():
        if "source" in metadata:
            existing_sources.add(metadata["source"])
    
    # Filter new PDFs
    new_pdfs = [pdf for pdf in pdf_files if os.path.basename(pdf) not in existing_sources]
    
    if not new_pdfs:
        return {"status": "skipped", "message": "All PDFs already in vector store", "files_processed": 0}
    
    print(f"Processing {len(new_pdfs)} new PDF(s)...")
    
    total_chunks_added = 0
    
    # Process each new PDF
    for pdf_path in new_pdfs:
        print(f"\nProcessing: {os.path.basename(pdf_path)}")
        
        try:
            # Extract text
            selected_pages = list(range(0, 100))  # Adjust as needed
            temp_output = f"{os.path.basename(pdf_path)}.txt"
            extract_text_from_pdf(pdf_path, selected_pages, temp_output)
            
            # Load and clean
            loader = TextLoader(temp_output, encoding="utf-8")
            new_docs = loader.load()
            
            # Add metadata and clean
            for doc in new_docs:
                doc.metadata["source"] = os.path.basename(pdf_path)
                doc.page_content = clean_text_for_bge(doc.page_content)
            
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
                metadatas = [
                    {
                        "content": chunk.page_content,
                        "source": os.path.basename(pdf_path),
                        "author": chunk.metadata.get("author", ""),
                        "year": chunk.metadata.get("year", ""),
                        "description": chunk.metadata.get("description", ""),
                        "chunk_hash": get_chunk_hash(chunk.page_content)
                    }
                    for chunk in chunks_to_add
                ]
                vector_store.add_embeddings(embeddings, metadatas)
                total_chunks_added += len(chunks_to_add)
                print(f"Added {len(chunks_to_add)} chunks from {os.path.basename(pdf_path)}")
            else:
                print(f"No new chunks to add from {os.path.basename(pdf_path)}")
            
            # Clean up temp file
            if os.path.exists(temp_output):
                os.remove(temp_output)
                
        except Exception as e:
            print(f"Error processing {os.path.basename(pdf_path)}: {e}")
            continue
    
    return {
        "status": "success",
        "files_processed": len(new_pdfs),
        "total_chunks_added": total_chunks_added,
        "message": f"Processed {len(new_pdfs)} PDF(s) and added {total_chunks_added} chunks"
    }



