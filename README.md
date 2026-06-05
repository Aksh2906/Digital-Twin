# Feynman Digital Twin

A conversational AI that talks like Richard Feynman. It pulls answers from his actual books, lectures, and interviews using a hybrid RAG pipeline, then responds in his speaking style with voice output.

## What it does

You type a question and the system does this behind the scenes:

1. Your question gets rewritten into better search queries by a fast LLM (Groq Llama 3.1)
2. Two search systems run in parallel: BM25 for keyword matching and ChromaDB for semantic similarity
3. Results from both are merged using Reciprocal Rank Fusion
4. A cross encoder reranks the top candidates and picks the best 4
5. Those 4 passages plus your chat history go to Gemini which generates a Feynman style response
6. If voice is enabled, XTTS v2 clones Feynman's voice from a 10 second reference clip and speaks the response

The system also tracks your interests and behavior over time and uses that to personalize responses.

## Features

| Category | Feature |
|----------|---------|
| Retrieval | Hybrid search (BM25 + semantic), RRF fusion, cross encoder reranking |
| Query | Automatic query refinement using Groq before retrieval |
| Generation | Gemini 2.5 Flash with Groq auto fallback, source quoting, conversational tone |
| Persona | Feynman speech patterns, temporal boundary (nothing after Feb 1988), filler words |
| Voice | XTTS v2 zero shot voice cloning from a 10 sec Feynman clip, interruptible playback |
| Sessions | Multiple conversations, switch between them, delete any, everything persists on disk |
| Profile | Auto detects your name, interests, knowledge level from your questions |
| UI | React SPA with sidebar, chat window, inspector panel showing sources and pipeline metrics |

## Tech Stack

| Component | What we use |
|-----------|------------|
| Server | FastAPI + Uvicorn |
| Frontend | React 18, Tailwind CSS, Babel standalone (single HTML file, no build step) |
| Primary LLM | Gemini 2.5 Flash |
| Fallback LLM | Groq Llama 3.1 8B (auto switches if Gemini errors) |
| Embeddings | BAAI/bge-small-en-v1.5 (local, free) |
| Vector DB | ChromaDB (embedded, local) |
| Keyword search | BM25Okapi |
| Reranker | ms-marco-MiniLM-L-12-v2 (cross encoder) |
| TTS | XTTS v2 (zero shot voice cloning from 10 sec reference) |
| Chunking | SemanticChunker from LangChain |

## Data Sources

Text files in `data/raw/`:

- Surely You're Joking, Mr. Feynman (full book)
- The Feynman Lectures on Physics Vol. 1
- Feynman's Nobel Lecture
- Feynman Interviews (transcripts)

About 1.2 MB total, split into 575 semantic chunks.

## Project Structure

```
.
├── app.py                                       # FastAPI server and RAG pipeline
├── the_feynman_archive_digital_researcher.html   # React frontend
├── requirements.txt
├── .env                                         # API keys
│
├── src/
│   ├── retriever.py          # hybrid retrieval (BM25 + ChromaDB + RRF)
│   ├── query_refiner.py      # rewrites user queries using Groq
│   ├── llm.py                # Gemini + Groq with auto fallback
│   ├── ingest.py             # loads docs, chunks, builds indexes
│   ├── tts.py                # XTTS v2 voice cloning
│   ├── memory.py             # chat memory utils
│   ├── sessions.py           # multi session management
│   └── user_profile.py       # user profiling and auto analysis
│
├── data/
│   ├── raw/                  # source text files
│   └── feynman_voice_sample.wav  # 10 sec reference clip for voice cloning
│
├── chroma_db_2/              # vector database (generated)
├── bm25_index.pkl            # keyword index (generated)
└── memory/
    ├── sessions/             # saved conversations
    └── user_profile.json     # user profile
```

## How to run

1. Clone the repo

```
git clone https://github.com/your-username/Digital-Twin.git
cd Digital-Twin
```

2. Create a virtual environment

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

5. Run ingestion (only once, or when you add new source files)

```
python -m src.ingest
```

6. For voice cloning, add a 10 second WAV clip of Feynman speaking to `data/feynman_voice_sample.wav` (see `VOICE_CLONING_GUIDE.md` for how to get one)

7. Start the app

```
python app.py
```

8. Open `http://localhost:8501`

## How sessions work

Every conversation saves as a JSON file in `memory/sessions/`. Click "New Inquiry" for a fresh session. Old sessions stay in the sidebar and you can switch back anytime. Delete any session by hovering and clicking the X.

## How user profiling works

The system tracks what topics you ask about. Every 5 messages it sends your recent questions to Groq which figures out your name, interests, and knowledge level. This info goes into the system prompt so Feynman can personalize responses. You can also manually edit your profile in the Settings tab.

## How voice works

The app uses XTTS v2 for zero shot voice cloning. You provide one 10 to 15 second clip of Feynman speaking and XTTS extracts his voice characteristics and applies them to any generated text. Toggle "Feynman Voice" in the right panel to enable it.

The audio stops when you send a new message or switch sessions. It only plays once per response.

## Adding more source material

1. Put your new text file in `data/raw/`
2. Add a title mapping in `src/ingest.py` (the SOURCE_TITLES dict)
3. Run `python -m src.ingest` again
4. Restart the app

## Other docs

- `rec.md` has the full project documentation with architecture diagrams and design decisions
- `CHANGELOG_2026_06_05.md` has the detailed changelog
- `VOICE_CLONING_GUIDE.md` has instructions for setting up voice cloning
