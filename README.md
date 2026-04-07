
# ReefGuide 🪸

**ReefGuide** is an Al-based chat tool that is specialized in information about the Great Barrier Reef in Australia (Queensland). 
It was developed during the university class **Large Language Models & Retrieval-Augmented Generation: 
A Practice-Oriented Approach to Developing AI Applications** held by Elisabeth Steffen. 


## Features

- **RAG Pipeline:** uses semantic search over scientific documents 
- **Semantic chunking** of provided data sources
- **Intent-aware filtering:** to distinguish casual conversation from knowledge-seeking questions 
- **Modern UI:** with Streamlit, including chat history, chat clear-out function, source visibility, and a reef-themed design  


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

1. **Clone the Repository**
 ```bash
   git clone https://github.com/MerlindCPohl/rag_project_great_barrier_reef.git
   cd Code
   ```

2. **Create and Activate a Virtual Environment** (macOS)
```bash
   python3.10 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Python Dependencies**
 ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Install and Start Ollama**  
   Install from: https://ollama.com/download  
   Then pull the model:
   ```bash
   ollama pull llama3
   ```

5. **Run ReefGuide Interface 🪸**
   ```bash
   streamlit run app.py
   ```

6. **Open the Local URL**
   Click `http://localhost:8501` in your local terminal. 


## Screenshots of Example Usage

<img width="1440" height="772" alt="Bildschirmfoto 2026-04-07 um 19 14 56" src="https://github.com/user-attachments/assets/380eb3eb-61e0-42c6-8888-6a9d3f83a9b4" />


<img width="1440" height="768" alt="Bildschirmfoto 2026-04-07 um 20 21 19" src="https://github.com/user-attachments/assets/f7692fdd-f178-4681-927f-ce055ab71926" />


<img width="1440" height="767" alt="Bildschirmfoto 2026-04-07 um 20 23 54" src="https://github.com/user-attachments/assets/217bed4d-708c-433a-97a8-c0e3d7d9898b" />


## Dislaimer

Note that ReefGuide is an **AI-based tool** that generates answers to your questions.
While it aims to provide accurate information, **responses may contain errors or be incomplete.**
Please verify important information using reliable sources.
Please note that the Marine Park Authority branding is **fictional and for illustrative purposes only.** 
This project is **not** affiliated with, endorsed by, or connected to the actual Great Barrier Reef Marine Park Authority.


## Authors 🐠

Merlind C. Pohl

https://github.com/MerlindCPohl

