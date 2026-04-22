from src.rag_pipeline import load_documents, split_documents, build_vectorstore

print("Loading documents...")
docs = load_documents()

print(f"Loaded {len(docs)} pages")

print("Splitting into chunks...")
chunks = split_documents(docs)

print(f"Created {len(chunks)} chunks")

print("Building vector database...")
build_vectorstore(chunks)

print("DONE: Vector DB created successfully.")