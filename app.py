import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from sentence_transformers import CrossEncoder
from src.memory import load_memory, save_memory, summarize_old_memory
from src.llm import get_llm
import time
import os

load_dotenv()


st.set_page_config(
    page_title="Feynman Digital Twin",
    layout="centered"
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Special+Elite&family=IBM+Plex+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Mono', monospace;
    background-color: #0e0e0e;
    color: #e8e0d0;
}

.stChatMessage {
    background: #1a1a1a !important;
    border: 1px solid #2e2e2e;
    border-radius: 4px;
    padding: 12px !important;
}

.stChatInputContainer {
    border-top: 1px solid #2e2e2e !important;
}

h1 {
    font-family: 'Special Elite', cursive !important;
    color: #f0c040 !important;
    letter-spacing: 2px;
}

.metric-box {
    background: #1a1a1a;
    border: 1px solid #2e2e2e;
    border-radius: 4px;
    padding: 8px 16px;
    font-size: 12px;
    color: #888;
}
</style>
""", unsafe_allow_html=True)


st.markdown("# Richard Feynman")
st.markdown("<p style='color:#888; font-size:13px; margin-top:-16px;'>Digital Twin · Nobel Laureate · Curious Mind</p>", unsafe_allow_html=True)
st.divider()


@st.cache_resource(show_spinner="Loading Feynman's knowledge base...")
def load_resources():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="chroma_db_2", embedding_function=embeddings)
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 10})
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are Richard Feynman — Nobel Prize-winning physicist, bongo-playing, curious mind.
Speak exactly as Feynman would: enthusiastic, direct, simple analogies, genuine wonder.
Start with "Look," or "The thing is,". Use "damn" occasionally not always .
Never say you are an AI. Be honest when you don't know something.
Cite sources if you are communicating the content out of the retrieved context or the long-term memory. Use [source] after the relevant sentence.
Ground answers in this retrieved context from Feynman's actual work:

{context}

IMPORTANT: Keep every response under 50 words. Be punchy like Feynman."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question} (respond in 50 words or less)")
    ])
    return retriever, reranker, llm, prompt

retriever, reranker, llm, prompt = load_resources()


if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_memory()
if "metrics" not in st.session_state:
    st.session_state.metrics = {"retrieval_ms": 0, "llm_ms": 0, "total_ms": 0}


with st.sidebar:
    st.markdown("### ⚙️ Session")
    exchanges = len(st.session_state.chat_history) // 2
    st.markdown(f"**Exchanges remembered:** {exchanges}")

    st.markdown("### ⏱️ Last Response")
    m = st.session_state.metrics
    st.markdown(f"""
    <div class='metric-box'>
    Retrieval: {m['retrieval_ms']}ms<br>
    LLM: {m['llm_ms']}ms<br>
    Total: {m['total_ms']}ms
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📚 About")
    st.caption("Built with LangChain, ChromaDB, HuggingFace Embeddings, and Gemini 2.5 Flash. RAG + reranking + long-term memory.")

    if st.button("🗑️ Clear Memory"):
        st.session_state.chat_history = []
        save_memory([])
        st.rerun()


def rerank(question, docs, top_n=4):
    pairs = [(question, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), reverse=True)
    return [doc for _, doc in ranked[:top_n]]


for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content)


if user_input := st.chat_input("Ask Feynman anything..."):
    with st.chat_message("user"):
        st.write(user_input)

    total_start = time.time()


    t0 = time.time()
    docs = retriever.invoke(user_input)
    docs = rerank(user_input, docs, top_n=4)
    retrieval_ms = int((time.time() - t0) * 1000)
    context = "\n\n".join([d.page_content for d in docs])

    
    t0 = time.time()
    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "chat_history": summarize_old_memory(st.session_state.chat_history),
        "question": user_input
    })
    llm_ms = int((time.time() - t0) * 1000)
    total_ms = int((time.time() - total_start) * 1000)

    
    st.session_state.metrics = {
        "retrieval_ms": retrieval_ms,
        "llm_ms": llm_ms,
        "total_ms": total_ms
    }

    with st.chat_message("assistant"):
        st.write(response.content)

    
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=response.content))
    save_memory(st.session_state.chat_history)
    st.rerun()