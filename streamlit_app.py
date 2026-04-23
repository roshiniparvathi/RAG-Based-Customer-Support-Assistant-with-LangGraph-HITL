import streamlit as st
from app.rag_pipeline import build_vector_db, create_graph

st.set_page_config(page_title="RAG Support Bot", layout="wide")

st.title("📘 Customer Support Assistant (RAG + LangGraph + Groq)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Upload Knowledge Base PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ PDF uploaded and processed!")

    db = build_vector_db("temp.pdf")

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 6
        }
    )

    app = create_graph(retriever)

    query = st.text_input("💬 Ask your question")

    if st.button("Submit") and query:
        with st.spinner("Thinking..."):
            result = app.invoke({"query": query})

        st.session_state.chat_history.append({
            "question": query,
            "answer": result["answer"],
            "confidence": result["confidence"],
            "sources": result.get("sources", [])
        })

st.subheader("🧠 Chat History")

for chat in reversed(st.session_state.chat_history):
    st.markdown(f"**🧑 Question:** {chat['question']}")
    st.markdown(f"**🤖 Answer:** {chat['answer']}")
    st.markdown(f"**📊 Confidence:** {chat['confidence']}")

    if chat["sources"]:
        best = chat["sources"][0]

        st.markdown("**📚 Source:**")
        st.markdown(f"""
📄 {best['source']} (Chunk {best['chunk_id']})  
📝 {best['preview']}
""")

    st.markdown("---")