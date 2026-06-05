from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from sentence_transformers import CrossEncoder
from src.memory import summarize_old_memory
from src.llm import get_llm, get_refiner_llm
from src.retriever import HybridRetriever
from src.query_refiner import refine_query
from src.sessions import (
    create_session, list_sessions, get_session_messages,
    get_session_langchain_messages, save_session_message,
    set_active_session, delete_session
)
from src.user_profile import load_profile, save_profile, update_profile_from_chat, get_profile_context, auto_analyze_profile, merge_auto_analysis
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import time
import os
import base64

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()

hybrid_retriever = HybridRetriever()
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
llm = get_llm()
refiner_llm = get_refiner_llm()

PROMPT = ChatPromptTemplate.from_messages([
    ("system", """STRICTLY FORBIDDEN: You MUST NOT use "em dashes" (—), "hyphens" (-), or "en dashes" (–) anywhere in your response. This applies to ALL text, including quotes and metadata.
Replace any intended dashes with commas, parentheses, or rephrase sentences to avoid them entirely.
DO NOT break sentences in the middle using a dash.
If a quote contains a dash, rewrite it or omit that specific phrase. Do not reproduce the forbidden punctuation in any form."""),
    ("system", """You are Richard Feynman — Nobel Prize-winning physicist, bongo-playing, curious mind.
Speak exactly as Feynman would: enthusiastic, direct, simple analogies, genuine wonder.
Start with "Look," or "The thing is,". Use "damn" occasionally not always.
Never say you are an AI. Be honest when you don't know something.

TEMPORAL BOUNDARY: Your knowledge and experiences end on February 15, 1988 — the date of your death.
You have NO knowledge of anything that happened after this date.
If asked about post-1988 topics, respond honestly: "I don't know about that — sounds like something after my time."

{researcher_context}

Ground your answers in this retrieved context from Feynman's actual work:

{context}

When answering, quote the specific text you're drawing from using this format:
> "quoted text" — [Source Title]
Only quote when you're directly using information from the retrieved context.

IMPORTANT: Keep every response under 70 words. Be punchy like Feynman and IT IS. CONVOSERTATION NOT SHORT ANSWER LIKE LLM. SO PROVIDE CONVOSERTATION with filler words and make it natural and as a conversation."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question} (respond in 70 words or less)")
])


def rerank_docs(question, docs, top_n=4):
    if not docs:
        return []
    pairs = [(question, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), reverse=True)
    return [doc for _, doc in ranked[:top_n]]


@app.get("/")
async def serve_ui():
    return FileResponse(
        os.path.join(BASE_DIR, "the_feynman_archive_digital_researcher.html"),
        media_type="text/html"
    )


# ── Sessions ──

@app.get("/api/sessions")
async def api_list_sessions():
    sessions, active = list_sessions()
    return JSONResponse({"sessions": sessions, "active": active})


@app.post("/api/sessions/new")
async def api_new_session():
    session = create_session()
    return JSONResponse(session)


@app.get("/api/sessions/{sid}")
async def api_get_session(sid: str):
    set_active_session(sid)
    messages = get_session_messages(sid)
    return JSONResponse({"messages": messages})


@app.delete("/api/sessions/{sid}")
async def api_delete_session(sid: str):
    delete_session(sid)
    return JSONResponse({"status": "deleted"})


@app.post("/api/sessions/{sid}/chat")
async def api_chat(sid: str, request: Request):
    body = await request.json()
    question = body.get("question", "")
    tts_enabled = body.get("tts", False)

    profile = load_profile()
    profile = update_profile_from_chat(question, profile)
    researcher_context = get_profile_context(profile)

    chat_history = get_session_langchain_messages(sid)

    total_start = time.time()

    t0 = time.time()
    refined_queries = refine_query(question, chat_history, refiner_llm)
    refine_ms = int((time.time() - t0) * 1000)

    t0 = time.time()
    docs = hybrid_retriever.retrieve(refined_queries, final_k=10)
    docs = rerank_docs(question, docs, top_n=4)
    retrieval_ms = int((time.time() - t0) * 1000)

    sources = []
    context = ""
    for idx, doc in enumerate(docs, 1):
        title = doc.metadata.get("source_title", "Unknown Source")
        preview = doc.metadata.get("preview", doc.page_content[:80])
        context += f'[Source {idx}: {title}]\n"{doc.page_content}"\n\n'
        sources.append({"title": title, "preview": preview})

    t0 = time.time()
    chain = PROMPT | llm
    response = chain.invoke({
        "context": context,
        "researcher_context": researcher_context,
        "chat_history": summarize_old_memory(chat_history),
        "question": question
    })
    llm_ms = int((time.time() - t0) * 1000)
    total_ms = int((time.time() - total_start) * 1000)

    audio_b64 = None
    if tts_enabled:
        try:
            from src.tts import synthesize_speech
            audio_path = synthesize_speech(response.content)
            with open(audio_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode()
            os.unlink(audio_path)
        except Exception:
            pass

    save_session_message(sid, "human", question)
    save_session_message(sid, "ai", response.content)

    if profile.get("interaction_count", 0) % 5 == 0 and profile.get("interaction_count", 0) > 0:
        all_msgs = get_session_messages(sid)
        user_questions = [m["content"] for m in all_msgs if m["role"] == "human"]
        analysis = auto_analyze_profile(user_questions, refiner_llm)
        if analysis:
            profile = merge_auto_analysis(profile, analysis)

    return JSONResponse({
        "answer": response.content,
        "sources": sources,
        "audio": audio_b64,
        "metrics": {
            "refine_ms": refine_ms,
            "retrieval_ms": retrieval_ms,
            "llm_ms": llm_ms,
            "total_ms": total_ms
        }
    })


# ── User Profile ──

@app.get("/api/profile")
async def api_get_profile():
    return JSONResponse(load_profile())


@app.post("/api/profile")
async def api_update_profile(request: Request):
    body = await request.json()
    profile = load_profile()
    if "name" in body:
        profile["name"] = body["name"]
    if "interests" in body:
        profile["interests"] = body["interests"]
    if "knowledge_level" in body:
        profile["knowledge_level"] = body["knowledge_level"]
    if "behavior_notes" in body:
        profile["behavior_notes"] = body["behavior_notes"]
    save_profile(profile)
    return JSONResponse(profile)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8501)