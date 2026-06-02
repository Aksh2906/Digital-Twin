import json
import os
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage

MEMORY_FILE = "memory/long_term.json"

def load_memory():
    """Load past conversations from disk."""
    os.makedirs("memory", exist_ok=True)
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r") as f:
        data = json.load(f)
    # Convert back to LangChain message objects
    messages = []
    for msg in data:
        if msg["role"] == "human":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    return messages

def save_memory(chat_history):
    """Save current conversation to disk."""
    os.makedirs("memory", exist_ok=True)
    data = []
    for msg in chat_history:
        data.append({
            "role": "human" if isinstance(msg, HumanMessage) else "ai",
            "content": msg.content,
            "timestamp": datetime.now().isoformat()
        })
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=2)

def summarize_old_memory(chat_history, keep_last=10):
    """Keep only last N exchanges to avoid prompt getting too long."""
    return chat_history[-keep_last * 2:]  # each exchange = 2 messages