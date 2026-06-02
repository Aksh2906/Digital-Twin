import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

def get_llm():
    """Returns Gemini if available, falls back to Groq."""
    
    # Try Gemini first
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=gemini_key,
                temperature=0.9
            )
            # Quick test to verify it's working
            llm.invoke("hi")
            print("Using Gemini")
            return llm
        except Exception as e:
            print(f"⚠️ Gemini failed: {e}")
    
    # Fallback to Groq (Llama 3)
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                api_key=groq_key,
                temperature=0.9
            )
            print("Using Groq (Llama 3) as fallback")
            return llm
        except Exception as e:
            print(f"Groq failed: {e}")
    
    raise Exception("Both Gemini and Groq are unavailable.")