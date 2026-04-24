import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from src.utils import get_embeddings, get_llm
from src.app_utils.paths import DATA_DIR, CHROMA_DIR


# -----------------------------
# Query Intent Detection
# -----------------------------
def is_summary_query(query: str) -> bool:
    return any(
        keyword in query.lower()
        for keyword in ["summarize", "summary", "overview", "high level"]
    )


# -----------------------------
# Document Loading
# -----------------------------
def load_documents():
    loader = PyPDFDirectoryLoader(
        str(DATA_DIR),
        glob="**/*.pdf"   # only load PDFs recursively
    )
    docs = loader.load()
    return docs


# -----------------------------
# Vectorstore Management
# -----------------------------
def vectorstore_exists() -> bool:
    return CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir())


def get_or_create_vectorstore():
    if not vectorstore_exists():
        docs = load_documents()

        if not docs:
            raise ValueError("No documents found to build vectorstore.")

        chunks = split_documents(docs)
        return build_vectorstore(chunks)

    return load_vectorstore()


# -----------------------------
# Chunking
# -----------------------------
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    filtered_chunks = []

    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown_document")

        # ✅ Only keep PDF chunks
        if not source.lower().endswith(".pdf"):
            continue

        chunk.metadata["chunk_id"] = i
        chunk.metadata["source"] = source.replace("\\", "/")

        filtered_chunks.append(chunk)

    return filtered_chunks


# -----------------------------
# Vectorstore Build / Load
# -----------------------------
def build_vectorstore(chunks):
    embeddings = get_embeddings()

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR)
    )


def load_vectorstore():
    embeddings = get_embeddings()

    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings
    )


# -----------------------------
# Retriever (UPDATED)
# -----------------------------
def get_retriever(query: str, default_k: int = 4):
    vectorstore = get_or_create_vectorstore()

    k = 10 if is_summary_query(query) else default_k

    return vectorstore.as_retriever(search_kwargs={"k": k})


# -----------------------------
# Simple QA (UPDATED)
# -----------------------------
def ask_question(question: str):
    retriever = get_retriever(question)

    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])

    llm = get_llm()

    prompt = f"""
You are answering a question using retrieved document context.

If the question asks for a summary:
- Provide a high-level synthesis across ALL content.

If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}
"""

    return llm.invoke(prompt).content