import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from src.utils import get_embeddings, get_llm
from src.app_utils.paths import DATA_DIR, CHROMA_DIR

def load_documents():
    loader = PyPDFDirectoryLoader(str(DATA_DIR))
    docs = loader.load()
    return docs

def vectorstore_exists() -> bool:
    """
    Checks if Chroma DB exists and has data.
    """
    return CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir())

def get_or_create_vectorstore():
    """
    Loads vectorstore if it exists, otherwise rebuilds from documents.
    """
    if not vectorstore_exists():
        docs = load_documents()

        if not docs:
            raise ValueError("No documents found to build vectorstore.")

        chunks = split_documents(docs)
        return build_vectorstore(chunks)

    return load_vectorstore()


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        source = chunk.metadata.get("source", "unknown_document")
        chunk.metadata["source"] = source.replace("\\", "/")

    return chunks


def build_vectorstore(chunks):
    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR) 
    )

    return vectorstore


def load_vectorstore():
    embeddings = get_embeddings()

    return Chroma(
        persist_directory=str(CHROMA_DIR), 
        embedding_function=embeddings
    )


def get_retriever(k: int = 5):
    vectorstore = get_or_create_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": k})


def ask_question(question: str):
    vectorstore = get_or_create_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])

    llm = get_llm()

    prompt = f"""
    Answer ONLY using the context below.
    If the answer is not in the context, say "I don't know."

    Context:
    {context}

    Question:
    {question}
    """

    return llm.invoke(prompt).content