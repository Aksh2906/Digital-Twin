from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from src.memory import load_memory, save_memory, summarize_old_memory
import os
from src.llm import get_llm
import time


load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
# db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
db = Chroma(persist_directory="chroma_db_2", embedding_function=embeddings)
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 4})

# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     google_api_key=os.getenv("GEMINI_API_KEY"),
#     temperature=0.7
# )

llm = get_llm()

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Richard Feynman Nobel Prize-winning physicist, bongo-playing, curious mind.
Speak exactly as Feynman would: enthusiastic, direct, use simple analogies, show genuine wonder.
Start explanations with "Look," or "The thing is,". Use "damn" occasionally.
Never say you are an AI. Be honest when you don't know something.
     
Ground your answers in this retrieved context from Feynman's actual work:
     
{context}
IMPORTANT: Keep every response under 50 words. Be punchy like Feynman."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question} (respond in 50 words or less)")
])

# Load long-term memory from previous sessions
chat_history = load_memory()
if chat_history:
    print(f"Welcome back! Remembering {len(chat_history)//2} previous exchanges.\n")

def chat(question):
    global chat_history
    startTime = time.time()
    docs = retriever.invoke(question)
    endTime = time.time()
    print(f"Retrieved {len(docs)} relevant chunks in {endTime - startTime:.2f} seconds.")
    context = "\n\n".join([d.page_content for d in docs])
    
    startTime = time.time()
    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "chat_history": summarize_old_memory(chat_history),
        "question": question
    })
    endTime = time.time()
    print(f"LLM response generated in {endTime - startTime:.2f} seconds.")
    
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response.content))
    save_memory(chat_history)  # persist after every message
    
    return response.content

print("Feynman Digital Twin ready! Type 'quit' to exit.\n")
print("=" * 50)

from src.voice import voice_loop

mode = input("Mode? (text/voice): ").strip().lower()
if mode == "voice":
    voice_loop(chat)
else:
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            break
        if not user_input:
            continue
        print(f"\nFeynman: {chat(user_input)}")