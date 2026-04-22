from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from src.utils import get_embeddings, get_llm

PERSIST_DIR = "chroma_db"

def load_documents(data_dir="data"):
    loader = PyPDFDirectoryLoader(data_dir)
    docs = loader.load()
    return docs

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        # Always overwrite to guarantee consistency
        chunk.metadata["chunk_id"] = i

        # Normalize source path
        source = chunk.metadata.get("source", "unknown_document")
        chunk.metadata["source"] = source.replace("\\", "/")

    return chunks

def build_vectorstore(chunks):
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    return vectorstore

def load_vectorstore():
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

def get_retriever(k: int = 5):
    """
    Returns a retriever for the persisted Chroma vector store.
    This is used by agent tools and other services that need retrieval
    without directly handling vector store setup.
    """
    vectorstore = load_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": k})

def ask_question(question: str):
    vectorstore = load_vectorstore()
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