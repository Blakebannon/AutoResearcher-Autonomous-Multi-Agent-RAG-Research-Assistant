from langchain_core.tools import tool
from langchain_tavily import TavilySearch


def create_document_retriever_tool(retriever):
    """
    Create a custom document retriever tool that returns retrieved text
    along with source metadata for better citations.
    """

    @tool
    def document_retriever(query: str) -> str:
        """
        Search the user's uploaded documents and return relevant content
        with source and chunk metadata.
        """

        docs = retriever.invoke(query)

        if not docs:
            return "No relevant document evidence found."

        results = []

        for doc in docs:
            source = doc.metadata.get("source", "unknown_document")
            chunk_id = doc.metadata.get("chunk_id", "unknown_chunk")

            # 🔍 DEBUG (safe for logs, remove later if desired)
            print("RETRIEVED SOURCE:", source)

            # ✅ HARD FILTER: only allow PDFs
            if not str(source).lower().endswith(".pdf"):
                continue

            results.append(
                f"[SOURCE: {source} | CHUNK: {chunk_id}]\n{doc.page_content}"
            )

        # 🚨 If everything got filtered out
        if not results:
            return "No valid PDF-based evidence found in retrieval."

        return "\n\n".join(results)

    return document_retriever


def create_web_search_tool():
    """
    Create the Tavily web search tool for current or missing information.
    """
    return TavilySearch(
        max_results=3,
        topic="general",
        include_answer=True,
        search_depth="advanced",
    )


def get_tools(retriever):
    """
    Return the full toolset for the AutoResearcher agent.
    """
    return [
        create_document_retriever_tool(retriever),
        create_web_search_tool(),
    ]