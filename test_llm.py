from src.utils import get_llm

def main():
    llm = get_llm()
    response = llm.invoke("Say hello in one short sentence.")
    print("\n=== LLM RESPONSE ===\n")
    print(response)

if __name__ == "__main__":
    main()