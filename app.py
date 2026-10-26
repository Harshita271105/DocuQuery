
from pathlib import Path

import streamlit as st
from sentence_transformers import SentenceTransformer

from index_cache import load_or_build_index
from rag_pipeline import (
    PDF_PATH,
    MODEL_NAME,
    load_chunks,
    answer_question,
    retrieve_passages,
)

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="DocuQuery | Document Q&A",
    page_icon="📄",
    layout="wide",
)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("📄 DocuQuery")
st.subheader("AI-Powered Document Question Answering")

st.markdown(
    """
    Ask questions about the **ONGC Annual Report 2024–25**.
    DocuQuery searches relevant document passages and uses AI
    to generate answers grounded in the retrieved context.
    """
)

st.divider()


# --------------------------------------------------
# Load and cache document chunks
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def get_document_chunks(pdf_path):
    if not Path(pdf_path).exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}. "
            "Check that the source document exists."
        )

    return load_chunks(pdf_path)


# --------------------------------------------------
# Load and cache the embedding model
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return SentenceTransformer(MODEL_NAME)


# --------------------------------------------------
# Initialize the document and search index
# --------------------------------------------------
try:
    with st.status(
        "Preparing DocuQuery...",
        expanded=True,
    ) as status:
        st.write("Step 1/2: Loading the embedding model...")
        model = get_embedding_model()

        st.write("Step 2/2: Loading the saved document index...")
        chunks, index = load_or_build_index(
            PDF_PATH,
            MODEL_NAME,
            model,
        )

        status.update(
            label="DocuQuery is ready!",
            state="complete",
            expanded=False,
        )
except Exception as error:
    st.error(
        "DocuQuery could not initialize. "
        "Check the PDF path, dependencies, and index cache."
    )
    with st.expander("Technical error details"):
        st.exception(error)

    st.stop()


# --------------------------------------------------
# Document overview
# --------------------------------------------------
st.markdown("### Document Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Document pages", "684")

with col2:
    st.metric("Searchable chunks", f"{len(chunks):,}")

with col3:
    st.metric("Passages retrieved per question", "3")

st.divider()


# --------------------------------------------------
# Question input
# --------------------------------------------------
st.markdown("### 💬 Ask your document")

with st.form("question_form"):
    question = st.text_area(
        "Enter your question",
        placeholder=(
            "Example: What is ONGC's target year for "
            "achieving net-zero carbon emissions?"
        ),
        height=100,
        max_chars=2000,
    )

    submitted = st.form_submit_button(
        "🔍 Find Answer",
        type="primary",
        use_container_width=True,
    )


# --------------------------------------------------
# Retrieve passages and generate an answer
# --------------------------------------------------

# --------------------------------------------------
# Retrieve passages and generate an answer
# --------------------------------------------------
if submitted:
    cleaned_question = question.strip()

    if not cleaned_question:
        st.warning("Please enter a question first.")

    else:
        try:
            with st.spinner(
                "Searching the document and generating your answer..."
            ):
                result = answer_question(
                    cleaned_question,
                    chunks,
                    model,
                    index,
                )

            st.divider()
            st.markdown("### 📝 Answer")
            st.write(result["answer"])

            st.markdown("### 📚 Source Passages")

            sources = result.get("sources", [])

            if sources:
                st.caption(
                    "Review these passages to verify the answer "
                    "against the original annual report."
                )

                for number, source in enumerate(sources, start=1):
                    page_number = source.get("page_number", "Unknown")
                    source_text = source.get("text", "").strip()
                    score = source.get("score")

                    label = f"Source {number} · PDF page {page_number}"

                    if score is not None:
                        label += f" · Similarity {score:.3f}"

                    with st.expander(label):
                        st.markdown(f"**PDF page: {page_number}**")

                        if score is not None:
                            st.caption(
                                f"Retrieval similarity score: {score:.3f}. "
                                "This is not a measure of answer correctness."
                            )

                        st.write(
                            source_text or "No passage text available."
                        )

            else:
                st.info(
                    "No source passages were returned. "
                    "Check the document for supporting evidence."
                )

        except Exception as error:
            st.error(
                "An error occurred while processing your question. "
                "Please try again."
            )

            with st.expander("Technical error details"):
                st.exception(error)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.header("ℹ️ About DocuQuery")

    st.write("**Source:** ONGC Annual Report 2024–25")
    st.write("**Embedding model:** `all-MiniLM-L6-v2`")
    st.write("**Vector search:** FAISS")
    st.write("**Answer generation:** Google Gemini")
    st.write("**Interface:** Streamlit")

    st.divider()

    st.markdown("**How it works**")
    st.caption(
        "1. Extract text from the PDF\n\n"
        "2. Convert text into searchable chunks\n\n"
        "3. Retrieve relevant passages\n\n"
        "4. Generate an answer using Gemini\n\n"
        "5. Review the source passages"
    )

    st.divider()

    st.caption(
        "AI-generated answers can contain errors. "
        "Verify important claims using the displayed source passages "
        "and the original document."
    )
