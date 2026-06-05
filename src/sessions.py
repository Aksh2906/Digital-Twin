import json
import os
import uuid
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage

SESSIONS_DIR = "memory/sessions"
SESSIONS_INDEX = "memory/sessions_index.json"


def _ensure_dirs():
    os.makedirs(SESSIONS_DIR, exist_ok=True)


def _load_index():
    _ensure_dirs()
    if not os.path.exists(SESSIONS_INDEX):
        return {"sessions": [], "active": None}
    with open(SESSIONS_INDEX, "r") as f:
        return json.load(f)


def _save_index(index):
    _ensure_dirs()
    with open(SESSIONS_INDEX, "w") as f:
        json.dump(index, f, indent=2)


def create_session(title="New Inquiry"):
    sid = str(uuid.uuid4())[:8]
    session = {
        "id": sid,
        "title": title,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    path = os.path.join(SESSIONS_DIR, f"{sid}.json")
    with open(path, "w") as f:
        json.dump({"messages": []}, f)

    index = _load_index()
    index["sessions"].insert(0, session)
    index["active"] = sid
    _save_index(index)
    return session


def list_sessions():
    index = _load_index()
    return index["sessions"], index.get("active")


def get_session_messages(sid):
    path = os.path.join(SESSIONS_DIR, f"{sid}.json")
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("messages", [])


def get_session_langchain_messages(sid):
    raw = get_session_messages(sid)
    messages = []
    for msg in raw:
        if msg["role"] == "human":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    return messages


def save_session_message(sid, role, content):
    _ensure_dirs()
    path = os.path.join(SESSIONS_DIR, f"{sid}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
    else:
        data = {"messages": []}

    data["messages"].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    index = _load_index()
    for s in index["sessions"]:
        if s["id"] == sid:
            if len(data["messages"]) == 1 and role == "human":
                s["title"] = content[:50] + ("..." if len(content) > 50 else "")
            s["updated_at"] = datetime.now().isoformat()
            break
    _save_index(index)


def set_active_session(sid):
    index = _load_index()
    index["active"] = sid
    _save_index(index)


def delete_session(sid):
    path = os.path.join(SESSIONS_DIR, f"{sid}.json")
    if os.path.exists(path):
        os.remove(path)
    index = _load_index()
    index["sessions"] = [s for s in index["sessions"] if s["id"] != sid]
    if index.get("active") == sid:
        index["active"] = index["sessions"][0]["id"] if index["sessions"] else None
    _save_index(index)
