from typing import TypedDict, List

from langchain_community.document_loaders import PyPDFLoader
import re
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langgraph.graph import StateGraph
from groq import Groq
from app.config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)


# =========================
# 📄 LOAD + SMART CHUNKING
# =========================
def split_into_sections(docs):
    full_text = "\n".join([d.page_content for d in docs])

    # Split by numbered sections (1., 2., etc.)
    sections = re.split(r'\n\d+\.\s', full_text)

    structured_docs = []

    for i, section in enumerate(sections):
        section = section.strip()

        if not section:
            continue

        structured_docs.append(
            Document(
                page_content=section,
                metadata={
                    "source": "knowledge_base.pdf",
                    "chunk_id": i
                }
            )
        )

    return structured_docs

def build_vector_db(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    chunks = split_into_sections(docs)

    embeddings = HuggingFaceEmbeddings()
    db = Chroma.from_documents(chunks, embeddings)

    return db


# =========================
# 🧠 STATE
# =========================
class GraphState(TypedDict):
    query: str
    context: List[str]
    answer: str
    confidence: float
    route: str
    sources: List[dict]


# =========================
# ⚙️ GRAPH
# =========================
def create_graph(retriever):

    def process(state: GraphState):
        query = state["query"]

        docs = retriever.invoke(query)

        context = [d.page_content for d in docs]

        sources = []

        for d in docs:
            text = d.page_content.replace("\n", " ").strip()

            # Skip very short or useless chunks
            if len(text) < 40:
                continue

            # Sentence preview
            sentences = text.split(".")
            preview = sentences[0].strip() + "."

            # relevance score
            score = sum(
                word in text.lower()
                for word in query.lower().split()
                if len(word) > 3
            )

            sources.append({
                "source": d.metadata.get("source", "unknown"),
                "chunk_id": d.metadata.get("chunk_id", "N/A"),
                "preview": preview,
                "score": score
            })

        # Sort best first
        sources = sorted(sources, key=lambda x: x["score"], reverse=True)

        prompt = f"""
You are a customer support assistant.

Answer the question using ONLY the context below.

IMPORTANT:
- If the question has multiple parts, answer ALL parts.
- Combine information from multiple sections if needed.
- Do NOT ignore any part of the question.
- If part of answer is missing, say what is available.

Context:
{context}

Question: {query}
"""

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"Groq API Error: {str(e)}"

        confidence = 0.7 if len(sources) >= 1 else 0.2

        return {
            "context": context,
            "answer": answer,
            "confidence": confidence,
            "sources": sources
        }

    def route(state: GraphState):
        if len(state["context"]) == 0 or state["confidence"] < 0.4:
            return {"route": "HITL"}
        return {"route": "ANSWER"}

    def hitl(state: GraphState):
        return {
            "answer": "⚠️ Escalated to human agent. Please wait for support.",
            "sources": []
        }

    graph = StateGraph(GraphState)

    graph.add_node("process", process)
    graph.add_node("route", route)
    graph.add_node("hitl", hitl)

    graph.set_entry_point("process")
    graph.add_edge("process", "route")

    graph.add_conditional_edges(
        "route",
        lambda s: s["route"],
        {
            "ANSWER": "__end__",
            "HITL": "hitl"
        }
    )

    graph.add_edge("hitl", "__end__")

    return graph.compile()
