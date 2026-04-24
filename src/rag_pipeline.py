import os
import shutil

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
        for keyword in [
            "summarize",
            "summary",
            "overview",
            "high level",
            "high-level",
            "main points",
            "key points",
            "what is this document about",
        ]
    )


# -----------------------------
# Document Loading
# -----------------------------
def load_documents():
    print("DATA_DIR:", DATA_DIR)
    print("DATA_DIR EXISTS:", DATA_DIR.exists())

    pdf_files = list(DATA_DIR.glob("**/*.pdf"))
    print("PDF FILES FOUND:", pdf_files)

    if not DATA_DIR.exists():
        raise ValueError(f"DATA_DIR does not exist: {DATA_DIR}")

    if not pdf_files:
        print("WARNING: No PDF files found in DATA_DIR.")
        return []

    loader = PyPDFDirectoryLoader(
        str(DATA_DIR),
        glob="**/*.pdf",
    )

    docs = loader.load()

    print("LOADED DOC COUNT:", len(docs))

    for i, doc in enumerate(docs[:3]):
        print(f"LOADED DOC {i + 1} SOURCE:", doc.metadata.get("source"))
        print(f"LOADED DOC {i + 1} PAGE:", doc.metadata.get("page"))
        print(f"LOADED DOC {i + 1} PREVIEW:", doc.page_content[:300])

    return docs


# -----------------------------
# Chunking
# -----------------------------
def split_documents(docs):
    print("DOCS BEFORE SPLITTING:", len(docs))

    if not docs:
        print("WARNING: No docs passed to split_documents().")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.split_documents(docs)

    print("RAW CHUNK COUNT:", len(chunks))

    filtered_chunks = []

    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown_document")
        normalized_source = str(source).replace("\\", "/")

        print("SOURCE BEFORE FILTER:", normalized_source)

        # Safer than endswith(".pdf") because some loaders may append paths oddly
        if ".pdf" not in normalized_source.lower():
            print("SKIPPING NON-PDF SOURCE:", normalized_source)
            continue

        if not chunk.page_content or not chunk.page_content.strip():
            print("SKIPPING EMPTY CHUNK FROM:", normalized_source)
            continue

        chunk.metadata["chunk_id"] = i
        chunk.metadata["source"] = normalized_source

        filtered_chunks.append(chunk)

    print("FILTERED CHUNK COUNT:", len(filtered_chunks))

    for i, chunk in enumerate(filtered_chunks[:3]):
        print(f"FILTERED CHUNK {i + 1} SOURCE:", chunk.metadata.get("source"))
        print(f"FILTERED CHUNK {i + 1} PAGE:", chunk.metadata.get("page"))
        print(f"FILTERED CHUNK {i + 1} PREVIEW:", chunk.page_content[:300])

    return filtered_chunks


# -----------------------------
# Vectorstore Management
# -----------------------------
def vectorstore_exists() -> bool:
    exists = CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir())
    print("CHROMA_DIR:", CHROMA_DIR)
    print("CHROMA_DIR EXISTS:", CHROMA_DIR.exists())
    print("VECTORSTORE FILES EXIST:", exists)
    return exists


def get_vectorstore_count(vectorstore) -> int:
    try:
        return vectorstore._collection.count()
    except Exception as e:
        print("WARNING: Could not read vectorstore count:", e)
        return 0


def clear_vectorstore():
    if CHROMA_DIR.exists():
        print("CLEARING EXISTING CHROMA_DIR:", CHROMA_DIR)
        shutil.rmtree(CHROMA_DIR)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def get_or_create_vectorstore(force_rebuild: bool = False):
    """
    Load existing vectorstore if it has documents.
    Rebuild if missing, empty, or force_rebuild=True.
    """

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    if force_rebuild:
        print("FORCE REBUILD ENABLED.")
        clear_vectorstore()

    if vectorstore_exists() and not force_rebuild:
        vectorstore = load_vectorstore()
        count = get_vectorstore_count(vectorstore)

        print("EXISTING VECTORSTORE COUNT:", count)

        if count > 0:
            return vectorstore

        print("Existing vectorstore is empty. Rebuilding...")
        clear_vectorstore()

    docs = load_documents()

    if not docs:
        raise ValueError(
            "No documents found to build vectorstore. "
            "Upload PDFs and make sure they are saved into DATA_DIR."
        )

    chunks = split_documents(docs)

    if not chunks:
        raise ValueError(
            "Documents were loaded, but no chunks were created. "
            "Check PDF text extraction, metadata source values, and filtering."
        )

    return build_vectorstore(chunks)


# -----------------------------
# Vectorstore Build / Load
# -----------------------------
def build_vectorstore(chunks):
    print("BUILDING VECTORSTORE WITH:", len(chunks), "chunks")

    if not chunks:
        raise ValueError("Cannot build vectorstore with zero chunks.")

    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    count = get_vectorstore_count(vectorstore)
    print("VECTORSTORE COUNT AFTER BUILD:", count)

    if count == 0:
        raise ValueError(
            "Vectorstore was built, but Chroma count is 0. "
            "This means documents were not written correctly."
        )

    return vectorstore


def load_vectorstore():
    print("LOADING EXISTING VECTORSTORE FROM:", CHROMA_DIR)

    embeddings = get_embeddings()

    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    print("LOADED VECTORSTORE COUNT:", get_vectorstore_count(vectorstore))

    return vectorstore


# -----------------------------
# Retriever
# -----------------------------
def get_retriever(query: str, default_k: int = 4):
    vectorstore = get_or_create_vectorstore()

    k = 10 if is_summary_query(query) else default_k

    print("RETRIEVER QUERY:", query)
    print("RETRIEVER K:", k)

    return vectorstore.as_retriever(search_kwargs={"k": k})


def retrieve_documents(query: str, default_k: int = 4):
    """
    Helper function for debugging retrieval directly.
    """
    retriever = get_retriever(query, default_k=default_k)
    docs = retriever.invoke(query)

    print("RETRIEVED DOC COUNT:", len(docs))

    for i, doc in enumerate(docs):
        print(f"RETRIEVED DOC {i + 1}")
        print("SOURCE:", doc.metadata.get("source"))
        print("PAGE:", doc.metadata.get("page"))
        print("CHUNK ID:", doc.metadata.get("chunk_id"))
        print("CONTENT PREVIEW:", doc.page_content[:500])

    return docs


# -----------------------------
# Simple QA
# -----------------------------
def ask_question(question: str):
    docs = retrieve_documents(question)

    context_blocks = []

    for doc in docs:
        source = doc.metadata.get("source", "unknown_source")
        page = doc.metadata.get("page", "unknown_page")
        chunk_id = doc.metadata.get("chunk_id", "unknown_chunk")

        context_blocks.append(
            f"[Source: {source}, Page: {page}, Chunk: {chunk_id}]\n"
            f"{doc.page_content}"
        )

    context = "\n\n---\n\n".join(context_blocks)

    llm = get_llm()

    prompt = f"""
You are answering a question using retrieved document context.

Rules:
- Use only the provided context.
- If the question asks for a summary, provide a high-level synthesis across the retrieved content.
- Include source/page references when useful.
- If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question:
{question}
"""

    return llm.invoke(prompt).content