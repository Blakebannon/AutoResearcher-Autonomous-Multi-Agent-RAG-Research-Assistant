from src.rag_pipeline import ask_question

response = ask_question("What is this document about?")
print("\n=== ANSWER ===\n")
print(response)