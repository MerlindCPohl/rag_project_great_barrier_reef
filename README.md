
# ReefGuide 🪸

**ReefGuide** is an Al-based chat tool that is specialized in information about the Great Barrier Reef in Australia (Queensland). 
It was developed during the university class **Large Language Models & Retrieval-Augmented Generation: 
A Practice-Oriented Approach to Developing AI Applications** at HTW Berlin, held by Elisabeth Steffen. 


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
├──.streamlit                      
│   ├── config.toml                 # Streamlit configuration
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
├── prompts.yaml                    # Prompt configuration and direct UI messages
├── requirements.txt                # Python dependencies
└── style.css                       # Streamlit styling    
```

## Prerequisites

- Python 3.10+ (Download: [python.org](https://www.python.org/))
  
- Ollama (Download: [ollama.com](https://ollama.com/download  ))


## Installation (Local Setup)

1. **Clone the Repository**
 ```bash
   git clone https://github.com/MerlindCPohl/rag_project_great_barrier_reef.git
   cd rag_project_great_barrier_reef
   ```

2. **Create and Activate a Virtual Environment** (macOS)
```bash
   python3 -m venv .venv

   source .venv/bin/activate   # macOS/Linux
   .\.venv\Scripts\activate    # Windows
   ```

3. **Install Python Dependencies**
 ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Start Ollama**  
 ```bash
   ollama pull llama3
 ```

5. **Run ReefGuide Interface 🪸**
 ```bash
   streamlit run app.py
 ```

6. **Open the Local URL**
   Click `http://localhost:8501` in your local terminal. 


## Configuration

The system's behavior can be adjusted in the following files:


### config.yaml (Core RAG Logic)

- **ingestion: breakpoint_threshold_amount**: Controls semantic chunking (higher = bigger chunks, lower = more granular segments).

- **embedding: model_name**: Model used to vectorize text (currently "BAAI/bge-m3"). 

- **retrieval: top_k**: Number of relevant document snippets retrieved from vector store (default = 5). 

- **retrieval: score_threshold**: Minimum similarity score required for a document to be considered (currently 0.3). Prevents from using irrelevant information.

- **llm: model**: The specific local model used via Ollama (currently llama3).

- **llm: temperature**: Controls the "creativity" of the response (higher values allow more varied phrasing).


### prompts.yaml (Prompts and direct UI Messages)

- **system_prompts**: Instructions for classification (reef vs. off-topic) and answer_generation (AI behaviour).

- **greetings**: Contains a keyword list to detect casual conversation.

- **messages**: Predefined responses.


### .streamlit/config.toml (UI Theme)
Primary Color & Backgrounds:
Customizes the reef-themed colors.


## Screenshots of Example Usage

<img width="1440" height="772" alt="Bildschirmfoto 2026-04-07 um 19 14 56" src="https://github.com/user-attachments/assets/380eb3eb-61e0-42c6-8888-6a9d3f83a9b4" />


<img width="1440" height="768" alt="Bildschirmfoto 2026-04-07 um 20 21 19" src="https://github.com/user-attachments/assets/f7692fdd-f178-4681-927f-ce055ab71926" />


<img width="1440" height="767" alt="Bildschirmfoto 2026-04-07 um 20 23 54" src="https://github.com/user-attachments/assets/217bed4d-708c-433a-97a8-c0e3d7d9898b" />


## Disclaimer

Note that ReefGuide is an **AI-based tool** that generates answers to your questions.
While it aims to provide accurate information, **responses may contain errors or be incomplete.**
Please verify important information using reliable sources.
Please note that the Marine Park Authority branding is **fictional and for illustrative purposes only.** 
This project is **not** affiliated with, endorsed by, or connected to the actual Great Barrier Reef Marine Park Authority.


## Authors 🐠

Merlind C. Pohl

https://github.com/MerlindCPohl

