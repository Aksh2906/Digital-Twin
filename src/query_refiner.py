from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import json

REFINE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a search query optimizer. Given a user's conversational question and recent chat context,
rewrite it into 1-3 precise search queries optimized for retrieving documents about Richard Feynman's life, work, and ideas.

Rules:
- Resolve pronouns and vague references using the chat context
- Expand vague questions into specific, searchable queries
- Include "Feynman" in at least one query
- Return ONLY a JSON array of query strings, nothing else
- Do NOT wrap in markdown code blocks

Example:
Chat context: User asked about teaching, Feynman talked about Brazil
User question: "what about that?"
Output: ["Feynman teaching experience Brazil", "Feynman university lectures Brazil"]

Example:
Chat context: No prior context.
User question: "tell me about the bongo thing"
Output: ["Feynman playing bongo drums", "Feynman percussion music hobby", "Feynman bongo stories"]"""),
    ("human", """Recent chat:
{chat_context}

User's question: {question}

JSON array of search queries:""")
])


def refine_query(question, chat_history, llm):
    recent = chat_history[-6:] if len(chat_history) > 6 else chat_history
    context_parts = []
    for msg in recent:
        role = "User" if isinstance(msg, HumanMessage) else "Feynman"
        context_parts.append(f"{role}: {msg.content}")
    chat_context = "\n".join(context_parts) if context_parts else "No prior context."

    try:
        chain = REFINE_PROMPT | llm
        response = chain.invoke({
            "chat_context": chat_context,
            "question": question
        })

        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        queries = json.loads(content)
        if isinstance(queries, list) and len(queries) > 0:
            if question not in queries:
                queries.append(question)
            return queries[:3]

    except Exception as e:
        print(f"Query refinement failed: {e}")

    return [question]
