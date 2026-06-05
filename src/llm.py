import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq


def get_llm():
    llms = []

    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        llms.append(ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=gemini_key,
            temperature=0.9
        ))

    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        llms.append(ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=groq_key,
            temperature=0.9
        ))

    if not llms:
        raise Exception("No LLM API keys found. Set GEMINI_API_KEY or GROQ_API_KEY in .env")

    if len(llms) == 1:
        return llms[0]

    return llms[0].with_fallbacks(llms[1:])


def get_refiner_llm():
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise Exception("GROQ_API_KEY required for query refinement.")
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=groq_key,
        temperature=0.0
    )