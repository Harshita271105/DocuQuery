
import streamlit as st
from pathlib import Path
from sentence_transformers import SentenceTransformer

from rag_pipeline import (
    PDF_PATH,
    MODEL_NAME,
    load_chunks,
    build_search_index,
    answer_question,
)

st.set_page_config(
    page_title="DocuQuery | Document Q&A",
    page_icon="📄",
    layout="wide",
)

st.title("📄 DocuQuery")
st.subheader("AI-powered Document Question Answering")
st.write(
    "Ask questions about your document and get answers grounded "
    "in its content, with source-page references."
)

@st.cache_resource
def initialize_pipeline():
    if not Path(PDF_PATH).exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    chunks = load_chunks(PDF_PATH)
    if not chunks:
        raise ValueError("No extractable text found in the PDF.")

    model = SentenceTransformer(MODEL_NAME)
    index = build_search_index(chunks, model)

    return chunks, model, index

try:
    with st.spinner("Loading document and preparing search..."):
        chunks, model, index = initialize_pipeline()

    st.success("Document is ready for questions.")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Pages in source PDF", "684")
    with col2:
        st.metric("Searchable text chunks", len(chunks))

    st.divider()
    question = st.text_area(
        "Ask a question about the ONGC annual report",
        placeholder="Example: What is ONGC's net-zero emissions target year?",
        height=100,
    )

    if st.button("Find Answer", type="primary"):
        if not question.strip():
            st.warning("Please enter a question first.")
        else:
            with st.spinner("Searching the document and generating an answer..."):
                result = answer_question(
                    question.strip(), chunks, model, index
                )

            st.markdown("### Answer")
            st.write(result["answer"])

            st.markdown("### Retrieved Sources")
            if result["sources"]:
                for i, source in enumerate(result["sources"], start=1):
                    with st.expander(
                        f"Source {i} — Page {source['page_number']}"
                    ):
                        st.write(source["text"])
            else:
                st.info("No relevant source passages were retrieved.")

    with st.sidebar:
        st.header("About DocuQuery")
        st.write("**Document:** ONGC Annual Report 2024–25")
        st.write("**Retrieval:** Sentence Transformers + FAISS")
        st.write("**Answer generation:** Google Gemini")
        st.caption(
            "Answers are generated from retrieved passages. "
            "Verify important information against the cited pages."
        )

except Exception as error:
    st.error(f"Could not initialize DocuQuery: {error}")
    st.info(
        "Check that the PDF exists, dependencies are installed, "
        "and the Gemini API configuration is valid."
    )
