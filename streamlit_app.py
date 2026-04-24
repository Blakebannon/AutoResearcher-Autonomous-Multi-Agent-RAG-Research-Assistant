import streamlit as st
from pathlib import Path

from src.rag_pipeline import load_documents, split_documents, build_vectorstore
from src.services.research_service import research


st.set_page_config(
    page_title="AutoResearcher",
    page_icon="📚",
    layout="wide",
)

st.title("📚 AutoResearcher")
st.caption("Autonomous multi-agent research assistant powered by RAG + LangGraph")

if "research_history" not in st.session_state:
    st.session_state.research_history = []


def build_markdown_report(result: dict) -> str:
    evaluation = result.get("evaluation", {})
    critic = result.get("critic", {})
    evidence = result.get("evidence", [])

    lines = []

    lines.append("# AutoResearcher Report\n")
    lines.append("## Final Answer\n")
    lines.append(result.get("final_answer") or result.get("answer", ""))

    lines.append("\n## Critic Review\n")
    lines.append(f"- Needs revision: {critic.get('needs_revision', False)}")
    lines.append(f"- Iteration count: {critic.get('iteration', 0)}")
    lines.append(f"- Feedback: {critic.get('critic_feedback', '')}")

    lines.append("\n## Judge Metrics\n")
    lines.append(f"- Groundedness: {evaluation.get('groundedness_score', 'N/A')}")
    lines.append(f"- Citation quality: {evaluation.get('citation_score', 'N/A')}")
    lines.append(f"- Completeness: {evaluation.get('completeness_score', 'N/A')}")
    lines.append(f"- Clarity: {evaluation.get('clarity_score', 'N/A')}")
    lines.append(f"- Overall: {evaluation.get('overall_score', 'N/A')}")
    lines.append(f"- Feedback: {evaluation.get('judge_feedback', '')}")

    lines.append("\n## Evidence Sources\n")

    for item in evidence:
        evidence_id = getattr(item, "evidence_id", "unknown")
        title = getattr(item, "title", "Untitled Source")
        source_type = getattr(item, "source_type", "unknown")
        relevance_score = getattr(item, "relevance_score", "N/A")
        url = getattr(item, "url", None)
        page = getattr(item, "page", None)
        chunk_id = getattr(item, "chunk_id", None)
        content = getattr(item, "content", "")

        lines.append(f"\n### [{evidence_id}] {title}")
        lines.append(f"- Source type: {source_type}")
        lines.append(f"- Relevance score: {relevance_score}")

        if url:
            lines.append(f"- URL: {url}")

        if page is not None:
            lines.append(f"- Page: {page}")

        if chunk_id is not None:
            lines.append(f"- Chunk: {chunk_id}")

        if content:
            lines.append("\nPreview:")
            lines.append(content[:1200])

    return "\n".join(lines)


def save_uploaded_files(uploaded_files, upload_dir: str = "data") -> list[str]:
    upload_path = Path(upload_dir)
    upload_path.mkdir(parents=True, exist_ok=True)

    saved_files = []

    for uploaded_file in uploaded_files:
        file_path = upload_path / uploaded_file.name

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        saved_files.append(str(file_path))

    return saved_files


def format_score(score):
    if score == "N/A" or score is None:
        return "N/A"

    try:
        return f"{float(score):.2f}"
    except (TypeError, ValueError):
        return score


def render_dashboard_summary(result: dict):
    evaluation = result.get("evaluation", {})
    critic = result.get("critic", {})
    evidence_items = result.get("evidence", [])

    st.success("Research workflow completed.")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Overall Score", format_score(evaluation.get("overall_score", "N/A")))

    with col2:
        st.metric("Evidence Items", len(evidence_items))

    with col3:
        st.metric("Revision Needed", str(critic.get("needs_revision", False)))

    with col4:
        st.metric("Iterations", critic.get("iteration", 0))


with st.sidebar:
    st.header("Document Indexing")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("Upload & Reindex"):
            with st.spinner("Saving uploaded files and rebuilding vector store..."):
                saved_files = save_uploaded_files(uploaded_files)

                docs = load_documents()
                chunks = split_documents(docs)
                build_vectorstore(chunks)

            st.success(f"Uploaded and indexed {len(saved_files)} file(s).")

            with st.expander("Uploaded files"):
                for file in saved_files:
                    st.write(file)

    st.divider()

    if st.button("Reindex Existing Documents in /data"):
        with st.spinner("Loading and indexing documents..."):
            docs = load_documents()
            chunks = split_documents(docs)
            build_vectorstore(chunks)
        st.success("Documents indexed successfully!")

    st.divider()
    st.caption("Workflow Process")
    st.write("Planner → Retrieval → Reranker → Researcher → Synthesizer → Critic → Judge")

    st.divider()
    st.subheader("Research History")

    if not st.session_state.research_history:
        st.caption("No research runs yet.")
    else:
        for i, item in enumerate(reversed(st.session_state.research_history), start=1):
            with st.expander(f"{i}. {item['query'][:50]}"):
                st.write(item["answer"][:500])
                st.caption(f"Overall score: {item['overall_score']}")


query = st.text_area(
    "Research question",
    placeholder="Ask a question about your documents, the web, or both...",
    height=120,
)

run_button = st.button("Run AutoResearcher", type="primary")

if run_button:
    if not query.strip():
        st.warning("Please enter a research question.")
        st.stop()

    with st.spinner("Running multi-agent research workflow..."):
        result = research(query)

    st.session_state.research_history.append(
        {
            "query": query,
            "answer": result.get("final_answer") or result.get("answer", ""),
            "overall_score": result.get("evaluation", {}).get("overall_score", "N/A"),
        }
    )

    if result.get("errors"):
        st.error("The workflow completed with errors.")
        with st.expander("View errors"):
            for error in result["errors"]:
                st.write(error)

    render_dashboard_summary(result)

    st.subheader("Final Answer")
    st.markdown(result.get("final_answer") or result.get("answer", ""))

    tab_answer, tab_sources, tab_critic, tab_judge, tab_trace = st.tabs(
        ["📝 Answer", "📚 Sources", "🧠 Critic Review", "📊 Judge Metrics", "🧭 Workflow Trace"]
    )

    with tab_answer:
        st.markdown(result.get("final_answer") or result.get("answer", ""))

    with tab_sources:
        evidence_items = result.get("evidence", [])

        if not evidence_items:
            st.info("No evidence returned.")
        else:
            st.write(f"Found {len(evidence_items)} evidence items.")

            for item in evidence_items:
                evidence_id = getattr(item, "evidence_id", "unknown")
                title = getattr(item, "title", "Untitled Source")
                source_type = getattr(item, "source_type", "unknown")
                relevance_score = getattr(item, "relevance_score", None)
                url = getattr(item, "url", None)
                page = getattr(item, "page", None)
                chunk_id = getattr(item, "chunk_id", None)
                content = getattr(item, "content", "")

                with st.expander(f"[{evidence_id}] {title}"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Source Type", source_type)

                    with col2:
                        if relevance_score is not None:
                            st.metric("Relevance", format_score(relevance_score))
                        else:
                            st.metric("Relevance", "N/A")

                    with col3:
                        if page is not None:
                            st.metric("Page", page)
                        elif chunk_id is not None:
                            st.metric("Chunk", chunk_id)
                        else:
                            st.metric("Location", "N/A")

                    if url:
                        st.markdown(f"[Open web source]({url})")

                    if content:
                        st.markdown("**Preview**")
                        st.write(content[:1500])

    with tab_critic:
        critic = result.get("critic", {})

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Needs Revision", str(critic.get("needs_revision", False)))

        with col2:
            st.metric("Iteration Count", critic.get("iteration", 0))

        st.markdown("**Critic Feedback**")
        st.write(critic.get("critic_feedback", "No critic feedback returned."))

    with tab_judge:
        evaluation = result.get("evaluation", {})

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Groundedness", format_score(evaluation.get("groundedness_score", "N/A")))

        with col2:
            st.metric("Citations", format_score(evaluation.get("citation_score", "N/A")))

        with col3:
            st.metric("Completeness", format_score(evaluation.get("completeness_score", "N/A")))

        with col4:
            st.metric("Clarity", format_score(evaluation.get("clarity_score", "N/A")))

        with col5:
            st.metric("Overall", format_score(evaluation.get("overall_score", "N/A")))

        st.markdown("**Judge Feedback**")
        st.write(evaluation.get("judge_feedback", "No judge feedback returned."))

    with tab_trace:
        st.markdown("**Planner Tasks**")
        st.json(result.get("tasks", []))

        st.markdown("**Route Log**")
        st.write(result.get("route_log", []))

        with st.expander("Raw State"):
            st.json(result.get("raw_state", {}), expanded=False)

    report = build_markdown_report(result)

    st.divider()

    st.download_button(
        label="Download Research Report",
        data=report,
        file_name="autoresearcher_report.md",
        mime="text/markdown",
    )