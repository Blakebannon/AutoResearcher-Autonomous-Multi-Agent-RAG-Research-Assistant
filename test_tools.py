from src.rag_pipeline import get_retriever
from src.tools.tools import get_tools

retriever = get_retriever()
tools = get_tools(retriever)

for tool in tools:
    print(tool.name)