# Feynman Digital Twin

A chatbot that talks like Richard Feynman. It uses RAG (Retrieval Augmented Generation) to answer questions based on Feynman's actual books, lectures, and interviews.

## What it does

You ask a question and the app searches through Feynman's writings, finds relevant chunks, reranks them, and sends them to an LLM which replies in Feynman's speaking style.

## Tech Stack

- Python
- Streamlit (frontend)
- LangChain (chaining everything together)
- ChromaDB (vector database)
- HuggingFace Embeddings (BGE-small-en-v1.5)
- Cross Encoder reranker (ms-marco-MiniLM-L-6-v2)
- Google Gemini 2.5 Flash (main LLM)
- Groq Llama 3.1 (fallback LLM)

## RAG Pipeline

1. **Ingestion** - text files from `data/raw/` are loaded using LangChain's DirectoryLoader
2. **Chunking** - documents are split using SemanticChunker (not fixed size, it splits based on meaning)
3. **Embedding** - chunks are embedded using HuggingFace BGE-small-en-v1.5 model
4. **Storing** - embeddings are stored in ChromaDB locally
5. **Retrieval** - when user asks a question, MMR search pulls top 10 similar chunks
6. **Reranking** - a CrossEncoder reranker picks the best 4 chunks out of 10
7. **Generation** - the top chunks + chat history go to Gemini/Groq which generates a Feynman-style answer

## Data Sources

- Feynman Lectures Vol 1
- Surely You're Joking Mr Feynman (book)
- Feynman's Nobel Lecture
- Feynman Interviews

All stored as .txt files in `data/raw/`.

## Project Structure

```
.
├── app.py                  # streamlit app (main entry)
├── src/
│   ├── ingest.py           # loads and chunks documents into ChromaDB
│   ├── chat.py             # terminal based chat (no UI)
│   ├── llm.py              # LLM setup with Gemini/Groq fallback
│   ├── memory.py           # saves/loads chat history to JSON
│   └── visualise_embeddings.py  # plots embeddings using TSNE
├── data/raw/               # source text files
├── chroma_db_2/            # vector database (auto generated)
├── memory/                 # long term chat memory (JSON)
└── .env                    # API keys
```

## How to run

1. Clone the repo
2. Create a virtual environment and activate it
```
python -m venv venv
source venv/bin/activate
```
3. Install dependencies
```
pip install -r requirements.txt
```
4. Add your API keys in `.env`
```
GEMINI_API_KEY=your_key
GROQ_API_KEY=your_key
```
5. Run ingestion first (only once)
```
python -m src.ingest
```
6. Run the app
```
streamlit run app.py
```

## Memory

The app saves chat history to `memory/long_term.json` so it remembers past conversations. It keeps last 10 exchanges in context to avoid the prompt getting too long.

## Extra

There is a script `src/visualise_embeddings.py` that plots all the stored embeddings in 2D using TSNE so you can see how the chunks cluster.
