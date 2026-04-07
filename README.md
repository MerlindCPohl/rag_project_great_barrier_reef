
# ReefGuide 🪸

**ReefGuide** is an Al-based chat tool that is specialized in information about the Great Barrier Reef in Australia (Queensland). 
It was developed during the university class **Large Language Models & Retrieval-Augmented Generation: 
A Practice-Oriented Approach to Developing AI Applications** held by Elisabeth Steffen. 


## Features

- **RAG Pipeline:** uses semantic search over scientific documents 
- **Semantic chunking** of provided data sources
- **Intent-aware filtering:** to distinguish casual conversation from knowledge-seeking questions 
- **Modern UI:** with Streamlit, including chat history, source visibility, and a reef-themed design  


## Tech Stack

**Frontend:** Streamlit

**LLM Orchestration:** LangChain & Ollama

**Vector Database:** FAISS

**Embeddings:** SentenceTransformers (Hugging Face)

**Language:** Python 3.10+


## Project Structure
```text
Code/
├──.streamlit                       # Streamlit configuration
│   ├── config.toml
├── assets/                         # UI Logo
├── data/
│   ├── evaluation/                 # Evaluation files
│   └── vector_store/               # Persisted FAISS index
│
├──src/
│   ├── __init__.py 
│   ├── embedding_manager.py        # Embedding model handling
│   ├── faiss_vector_store.py       # FAISS index management
│   ├── ingestion.py                # Document loading/chunking
│   ├── pipeline.py                 # Retrieval orchestration
│   ├── retriever.py                # Retrieval logic 
│   └── utils.py                    # Shared helper functions    
│         
├── app.py                          # Streamlit app     
├── config.yaml                     # Project/model configuration    
├── requirements.txt                # Python dependencies
└── style.css                       # Streamlit styling    
```

## Installation (Local Setup)

1. **Clone the repository**
 ```bash
   git clone https://github.com/MerlindCPohl/rag_project_great_barrier_reef.git
   cd Code
   ```

2. **Create and activate a virtual environment** (macOS)
```bash
   python3.10 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Python dependencies**
 ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Install and start Ollama**  
   Install from: https://ollama.com/download  
   Then pull the model:
   ```bash
   ollama pull llama3
   ```

5. **Run ReefGuide Interface 🪸**
   ```bash
   streamlit run app.py
   ```

6. **Open the local URL**
   Click `http://localhost:8501` in your local terminal. 


## Screenshots

![App Screenshot](https://dummyimage.com/468x300?text=App+Screenshot+Here)


## Authors 🐠

Merlind C. Pohl

https://github.com/MerlindCPohl

