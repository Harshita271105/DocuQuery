import streamlit as st
from index_cache import load_or_build_index
from pathlib import Path
from sentence_transformers import SentenceTransformer

from rag_pipeline import (
    PDF_PATH,
    MODEL_NAME,
    load_chunks,
    build_search_index,
    answer_question,
)
print("DOCUQUERY APP.PY STARTED", flush=True)
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
st.write(
    "Ask questions about the ONGC Annual Report 2024–25. "
    "Get answers grounded in document passages with source-page references."
)

st.divider()


# --------------------------------------------------
# Load and cache the document chunks
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def get_document_chunks(pdf_path):
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    return load_chunks(pdf_path)


# --------------------------------------------------
# Load and cache the embedding model
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return SentenceTransformer(MODEL_NAME)


# --------------------------------------------------
# Create and cache the FAISS index
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_search_index(_model, chunks):
    return build_search_index(chunks, _model)

print("REACHED PIPELINE INITIALIZATION", flush=True)
# --------------------------------------------------
# Initialize the application
# --------------------------------------------------

try:
    with st.status("Preparing DocuQuery...", expanded=True) as status:
        st.write("Step 1/3: Loading the embedding model...")
        model = get_embedding_model()

        st.write(
            "Step 2/3: Loading the saved index or building it if needed..."
        )
        chunks, index = load_or_build_index(
            PDF_PATH,
            MODEL_NAME,
            model,
        )

        st.write(f"Ready: {len(chunks):,} text chunks available.")

        status.update(
            label="DocuQuery is ready!",
            state="complete",
            expanded=False,
        )

except Exception as error:
    st.error("DocuQuery could not finish initializing.")
    st.exception(error)
    st.stop()


# --------------------------------------------------
# Document information
# --------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Document pages", "684")

with col2:
    st.metric("Searchable chunks", f"{len(chunks):,}")

with col3:
    st.metric("Retrieved passages per question", "3")

st.divider()


# --------------------------------------------------
# Question and answer interface
# --------------------------------------------------
st.markdown("### Ask your document")

with st.form("question_form"):
    question = st.text_area(
        "Your question",
        placeholder=(
            "Example: What is ONGC's target year "
            "for achieving net-zero carbon emissions?"
        ),
        height=100,
    )

    submitted = st.form_submit_button(
        "Find Answer",
        type="primary",
    )

if submitted:
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        try:
            with st.spinner(
                "Searching the document and generating your answer..."
            ):
                result = answer_question(
                    question.strip(),
                    chunks,
                    model,
                    index,
                )

            st.markdown("### Answer")
            st.write(result["answer"])

            st.markdown("### Source passages")

            sources = result.get("sources", [])

            if sources:
                for number, source in enumerate(sources, start=1):
                    page_number = source["page_number"]

                    with st.expander(
                        f"Source {number} — Page {page_number}"
                    ):
                        st.write(source["text"])
            else:
                st.info("No source passages were returned.")

        except Exception as error:
            st.error("An error occurred while answering your question.")
            st.exception(error)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.header("About DocuQuery")

    st.write("**Source:** ONGC Annual Report 2024–25")
    st.write("**Embeddings:** Sentence Transformers")
    st.write("**Vector search:** FAISS")
    st.write("**Answer generation:** Google Gemini")

    st.divider()

    st.caption(
        "Answers are grounded in retrieved document passages. "
        "Check the source pages when verifying important information."
    )
