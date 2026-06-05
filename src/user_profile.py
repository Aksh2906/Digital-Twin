import json
import os
from datetime import datetime

PROFILE_PATH = "memory/user_profile.json"

DEFAULT_PROFILE = {
    "name": "",
    "interests": [],
    "knowledge_level": "unknown",
    "interaction_count": 0,
    "common_topics": {},
    "behavior_notes": [],
    "created_at": None,
    "updated_at": None
}


def load_profile():
    os.makedirs("memory", exist_ok=True)
    if not os.path.exists(PROFILE_PATH):
        return dict(DEFAULT_PROFILE)
    with open(PROFILE_PATH, "r") as f:
        return json.load(f)


def save_profile(profile):
    os.makedirs("memory", exist_ok=True)
    profile["updated_at"] = datetime.now().isoformat()
    if not profile.get("created_at"):
        profile["created_at"] = profile["updated_at"]
    with open(PROFILE_PATH, "w") as f:
        json.dump(profile, f, indent=2)


def update_profile_from_chat(question, profile):
    profile["interaction_count"] = profile.get("interaction_count", 0) + 1

    keywords = ["quantum", "physics", "math", "feynman", "lecture", "science",
                "teaching", "curiosity", "experiment", "particle", "energy",
                "Nobel", "bongo", "Los Alamos", "Caltech", "MIT", "Brazil",
                "gravity", "light", "electron", "relativity", "thermodynamics"]

    topics = profile.get("common_topics", {})
    for kw in keywords:
        if kw.lower() in question.lower():
            topics[kw.lower()] = topics.get(kw.lower(), 0) + 1
    profile["common_topics"] = topics

    save_profile(profile)
    return profile


def auto_analyze_profile(questions_list, llm):
    """Use LLM to auto-detect user name, interests, knowledge level from their questions."""
    if len(questions_list) < 3:
        return None

    recent = questions_list[-15:]
    questions_text = "\n".join(f"- {q}" for q in recent)

    prompt = f"""Analyze these user questions asked to a Richard Feynman chatbot and extract user traits.
Return ONLY valid JSON, no markdown.

Questions:
{questions_text}

Return this exact JSON structure:
{{
  "name": "detected name or empty string if not mentioned",
  "interests": ["list of 3-5 detected interests"],
  "knowledge_level": "beginner/intermediate/advanced",
  "behavior_notes": ["1-2 short observations about how they ask questions"]
}}"""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(content)
    except Exception as e:
        print(f"Auto-analyze failed: {e}")
        return None


def merge_auto_analysis(profile, analysis):
    """Merge LLM analysis into existing profile without overwriting user edits."""
    if not analysis:
        return profile

    if analysis.get("name") and not profile.get("name"):
        profile["name"] = analysis["name"]

    if analysis.get("interests"):
        existing = set(profile.get("interests", []))
        for interest in analysis["interests"]:
            existing.add(interest)
        profile["interests"] = list(existing)[:8]

    if analysis.get("knowledge_level") and profile.get("knowledge_level", "unknown") == "unknown":
        profile["knowledge_level"] = analysis["knowledge_level"]

    if analysis.get("behavior_notes"):
        existing = profile.get("behavior_notes", [])
        for note in analysis["behavior_notes"]:
            if note not in existing:
                existing.append(note)
        profile["behavior_notes"] = existing[-5:]

    save_profile(profile)
    return profile


def get_profile_context(profile):
    parts = []
    if profile.get("name"):
        parts.append(f"The researcher's name is {profile['name']}.")
    if profile.get("interests"):
        parts.append(f"They are interested in: {', '.join(profile['interests'])}.")
    if profile.get("knowledge_level") and profile["knowledge_level"] != "unknown":
        parts.append(f"Their knowledge level is {profile['knowledge_level']}.")
    if profile.get("common_topics"):
        top = sorted(profile["common_topics"].items(), key=lambda x: x[1], reverse=True)[:5]
        if top:
            parts.append(f"They frequently ask about: {', '.join(t[0] for t in top)}.")
    if profile.get("behavior_notes"):
        recent = profile["behavior_notes"][-3:]
        parts.append(f"Notes: {'; '.join(recent)}.")
    return " ".join(parts) if parts else ""
