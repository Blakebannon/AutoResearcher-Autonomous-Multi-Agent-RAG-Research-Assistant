import streamlit as st
from src.rag_pipeline import load_documents, split_documents, build_vectorstore
from src.services.research_service import research

st.set_page_config(page_title="AutoResearcher", page_icon="📚")

st.title("📚 AutoResearcher – Week 1 Prototype")
st.caption("Basic RAG over your PDF documents")

if st.button("Index Documents in /data"):
    with st.spinner("Loading and indexing documents..."):
        docs = load_documents()
        chunks = split_documents(docs)
        build_vectorstore(chunks)
    st.success("Documents indexed successfully!")

question = st.text_input("Ask a question about your documents:")

if st.button("Get Answer") and question:
    with st.spinner("Researching..."):
        result = research(question)

    st.subheader("Answer")
    st.write(result["answer"])

    if result["tools_used"]:
        st.subheader("Tools Used")
        st.write(", ".join(result["tools_used"]))