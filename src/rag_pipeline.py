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
    return splitter.split_documents(docs)

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