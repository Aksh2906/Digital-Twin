# Changelog — June 5, 2026

## Feynman Digital Twin — Full Day Engineering Log

---

## 🏗️ Architecture Overhaul

### 1. RAG Pipeline Upgrade: Hybrid Search + Reranking

**What changed:**
- Replaced single ChromaDB semantic search with **Hybrid Retrieval** (BM25 + Semantic + RRF Fusion)
- Added **Query Refinement** before retrieval using Groq (Llama 3.1)
- Added **Cross-Encoder Reranking** using `ms-marco-MiniLM-L-12-v2`
- Added **source quoting** in responses (`> "quoted text" — [Source Title]`)
- Added **temporal boundary** (Feynman dies Feb 15, 1988 — no post-1988 knowledge)

**Files created/modified:**
| File | Action | Purpose |
|------|--------|---------|
| `src/retriever.py` | Rewritten | HybridRetriever class: BM25 + ChromaDB + RRF fusion + deduplication |
| `src/query_refiner.py` | Created | Rewrites vague user questions into 1-3 precise search queries via Groq |
| `src/ingest.py` | Modified | Now builds both ChromaDB and BM25 index (`bm25_index.pkl`), enriches metadata with `source_title`, `chunk_index`, `preview` |
| `src/llm.py` | Modified | Gemini 2.5 Flash primary, Groq Llama 3.1 auto-fallback, separate refiner LLM |

**Why Hybrid Search over pure Semantic:**
- Semantic search misses exact keyword matches. If user asks "Challenger disaster", BM25 finds documents containing those exact words instantly
- Semantic search finds conceptual matches ("space shuttle accident") but ranks them by embedding similarity which can be noisy
- RRF fusion merges both ranked lists without needing score normalization — a proven technique from the original RRF paper (k=60)

**Why NOT ColBERT or dense retrieval reranking:**
- ColBERT requires pre-computing token-level embeddings for all documents — heavy storage and setup
- Our corpus is small (575 chunks) — BM25 + semantic + cross-encoder reranking is more than sufficient
- Cross-encoder `ms-marco-MiniLM-L-12-v2` is fast enough for real-time use on this corpus size

**Why NOT LLM-based reranking (e.g., GPT reranker):**
- Adds 2-5 seconds latency per query (API call for each document pair)
- Cross-encoder achieves comparable quality for factual retrieval at 10-50ms
- Cost: LLM reranking burns API credits on every query

**Why query refinement with Groq instead of Gemini:**
- Speed: Groq inference is 3-5x faster than Gemini for short tasks
- Cost: Refinement is a cheap, mechanical task — don't need a frontier model
- Temperature=0 for deterministic query rewriting (no creativity needed here)

---

### 2. Frontend: Streamlit → FastAPI + Custom React UI

**What changed:**
- Replaced Streamlit (`streamlit run app.py`) with **FastAPI** serving a custom React single-page app
- The React UI is in `the_feynman_archive_digital_researcher.html` — a self-contained SPA with Tailwind CSS, particle effects, glassmorphism, and a research-lab aesthetic
- Backend exposes REST API endpoints; frontend calls them via `fetch()`

**API endpoints:**
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Serves the HTML UI |
| `/api/sessions` | GET | List all chat sessions |
| `/api/sessions/new` | POST | Create a new session |
| `/api/sessions/{sid}` | GET | Load a specific session's messages |
| `/api/sessions/{sid}` | DELETE | Delete a session |
| `/api/sessions/{sid}/chat` | POST | Send a message in a session (runs full RAG pipeline) |
| `/api/profile` | GET | Get user profile |
| `/api/profile` | POST | Update user profile |

**Why FastAPI over Streamlit:**
- Streamlit re-runs the entire script on every interaction — no way to render a custom React UI
- Streamlit's component model doesn't support the level of UI customization needed (sidebar sessions, inspector panel, particle effects)
- FastAPI is async, lightweight, and serves static HTML natively
- The user explicitly requested using their custom HTML template

**Why NOT Next.js / Vite / full React build:**
- The user wanted a single HTML file — no build step, no node_modules, no npm
- Babel standalone (v7) transpiles JSX in-browser — zero build toolchain
- For a single-page research tool, this is simpler to deploy and modify

**Why NOT Flask:**
- FastAPI has native async support — important for TTS (edge-tts is async)
- Auto-generates OpenAPI docs at `/docs` for free
- Type hints and Pydantic validation built-in

---

### 3. TTS: Coqui XTTS v2 → edge-tts (Microsoft Edge TTS)

**What changed:**
- Replaced `TTS` package (Coqui XTTS v2) with `edge-tts` (Microsoft Edge TTS)
- Voice: `en-US-RogerNeural` (American male, professor-like cadence)
- Output format: MP3 (was WAV)

**Why edge-tts over XTTS:**
- XTTS v2 requires `spacy` → `thinc` → needs specific C++ build tools
- `thinc` failed to build on Python 3.13 (the user's environment)
- edge-tts has zero native dependencies, works on any Python version
- No model download needed (XTTS downloads ~2GB on first run)
- Latency: edge-tts generates speech in ~1 second; XTTS takes 5-15 seconds on CPU

**Why NOT just skip TTS entirely:**
- The user specifically wanted Feynman to "speak" — voice output is a core feature
- edge-tts is a good placeholder until the user fine-tunes XTTS on Feynman's actual voice (see guide)

**Why NOT OpenAI TTS / ElevenLabs:**
- Costs money per character
- The user wants a local solution they can eventually fine-tune
- edge-tts is free and unlimited

---

### 4. Multi-Session Chat

**What changed:**
- Each conversation is stored as a separate JSON file in `memory/sessions/`
- Sessions index in `memory/sessions_index.json` tracks all sessions and the active one
- Sidebar shows all conversations — click to switch, X to delete
- "New Inquiry" creates a fresh session while preserving all previous ones
- Sessions auto-title from the first user message

**Files created:**
| File | Purpose |
|------|---------|
| `src/sessions.py` | CRUD operations: create, list, load, delete, save messages, set active |

**Why file-based storage over SQLite:**
- Simpler to debug (just open the JSON files)
- No additional dependencies
- For a single-user research tool, JSON files are perfectly adequate
- Easy to back up — just copy the `memory/` directory

**Why NOT a database (Postgres, Mongo):**
- Massive overkill for a single-user desktop app
- Adds deployment complexity
- The user can always migrate later — the session format is clean enough to import

---

### 5. User Profile & Auto-Analysis

**What changed:**
- Created `src/user_profile.py` for tracking user identity and behavior
- Profile stores: name, interests, knowledge level, interaction count, common topics, behavior notes
- **Auto-analysis**: every 5 interactions, the user's questions are sent to Groq LLM which extracts name, interests, knowledge level, and behavior patterns automatically
- Profile context is injected into the system prompt so Feynman can personalize responses
- Settings tab in sidebar shows and allows editing of all profile fields
- Profile auto-refreshes when switching to Settings tab

**Files created:**
| File | Purpose |
|------|---------|
| `src/user_profile.py` | Profile CRUD, keyword tracking, LLM-based auto-analysis, profile context generation |

**Why auto-detect instead of manual-only:**
- Users don't fill out profiles — they just want to chat
- LLM can infer interests from question patterns ("keeps asking about quantum mechanics → interested in quantum physics")
- Manual override still available for corrections

**Why every 5 interactions instead of every message:**
- LLM call adds latency — doing it on every message would slow down responses
- 5 interactions gives enough signal for meaningful analysis
- Profile changes are gradual, not per-message

---

### 6. Audio Replay Bug Fix

**What changed:**
- Moved from **per-message `<audio>` elements** to a **single global `Audio` object**
- Audio only plays once per message (tracked via `playedAudioIds` Set)
- Audio stops immediately when:
  - User switches to a different session
  - User sends a new message (interruption)
  - User creates a new session
- Old messages loaded from history are pre-marked as "already played" — no replay

**Why a global Audio object instead of per-message:**
- React re-renders components when state changes → per-message `<audio>` tags re-mount → audio restarts
- A single Audio object lives outside the React render cycle — immune to re-renders
- Easier to control (pause/stop/resume) from anywhere in the app

---

## 🧹 Code Cleanup

**What changed:**
- Removed all comments added by the AI from: `retriever.py`, `query_refiner.py`, `llm.py`, `ingest.py`, `tts.py`
- Kept only the user's original comments and essential docstrings
- Removed dead XTTS code from `tts.py` (commented-out block)

---

## 📝 Prompt Engineering (User Changes)

The user made these prompt modifications directly:

1. **Dash prohibition**: Added a system message strictly forbidding em dashes, hyphens, and en dashes in responses
2. **Word limit**: Reduced from 100 → 75 → 70 words
3. **Conversational tone**: Added instruction for filler words and natural conversation style instead of "short answer like LLM"

---

## 📊 Current File Structure

```
Digital-Twin/
├── app.py                          # FastAPI server (was Streamlit)
├── the_feynman_archive_*.html      # React SPA (full UI)
├── requirements.txt                # Updated deps
├── src/
│   ├── retriever.py                # Hybrid BM25 + Semantic + RRF
│   ├── query_refiner.py            # Groq-powered query rewriting
│   ├── llm.py                      # Gemini + Groq with auto-fallback
│   ├── ingest.py                   # Document ingestion pipeline
│   ├── tts.py                      # edge-tts (Microsoft Edge TTS)
│   ├── memory.py                   # Legacy memory (still used for summarize_old_memory)
│   ├── sessions.py                 # Multi-session CRUD
│   └── user_profile.py             # User profiling + auto-analysis
├── memory/
│   ├── sessions/                   # Per-session JSON files
│   ├── sessions_index.json         # Session listing
│   └── user_profile.json           # User identity & behavior
├── chroma_db_2/                    # ChromaDB vector store
├── bm25_index.pkl                  # BM25 keyword index
└── data/raw/                       # Source texts (4 files)
```

---

## 🚫 What Was NOT Done (and Why)

| Considered | Rejected Because |
|-----------|-----------------|
| **LangGraph agent** | Overkill — the pipeline is linear (refine → retrieve → rerank → generate). No branching or tool-calling needed. |
| **Streaming responses** | Adds complexity to both backend (SSE/WebSocket) and frontend. 70-word responses generate fast enough (~2s) that streaming provides minimal UX benefit. |
| **Vector DB migration to Pinecone/Weaviate** | Cloud dependency for a local-first tool. ChromaDB is embedded and free. |
| **Fine-tuning the LLM** | Requires significant compute, curated dataset, and time. Prompt engineering achieves 80% of the persona quality. Documented as a future step in the XTTS guide. |
| **Authentication / multi-user** | This is a single-user research tool. Adding auth would add complexity without value. |
| **Docker containerization** | User is running locally with a venv. Docker adds a layer of abstraction they don't need right now. |
| **WebSocket for real-time chat** | HTTP POST + JSON response is simpler and more debuggable. WebSockets matter for streaming or collaborative features — neither applies here. |
| **Embedding model upgrade to OpenAI/Cohere** | API cost on every query. `BAAI/bge-small-en-v1.5` is free, local, and strong enough for 575 chunks. |
| **RAG evaluation framework (RAGAS/DeepEval)** | Good practice but premature — the user is still iterating on features. Worth adding when the knowledge base grows. |
| **Conversation summarization for long context** | `summarize_old_memory()` already truncates to last 10 exchanges. Full summarization (via LLM) would add latency. Current approach is sufficient for 70-word responses. |

---

## 🔮 Recommended Next Steps

1. **Fine-tune XTTS v2** on Feynman's voice (guide provided in `xtts_persona_guide.md`)
2. **Expand knowledge base** — add more Feynman books to `data/raw/` and re-run `python -m src.ingest`
3. **Add streaming** if responses grow longer or you switch to a slower model
4. **Deploy** — wrap in Docker, host on a VPS, or use ngrok for quick sharing
