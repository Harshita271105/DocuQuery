
import hashlib
from pathlib import Path

import streamlit as st
from sentence_transformers import SentenceTransformer

from index_cache import load_or_build_index
from rag_pipeline import (
    PDF_PATH,
    MODEL_NAME,
    answer_question,
)

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="DocuQuery | Document Q&A",
    page_icon="📄",
    layout="wide",
)

DEFAULT_PDF_PATH = Path(PDF_PATH)
UPLOAD_DIR = Path("storage") / "uploads"
MAX_UPLOAD_SIZE_MB = 50

# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("📄 DocuQuery")
st.subheader("AI-Powered Document Question Answering")

st.markdown(
    """
    Upload a PDF or use the default **ONGC Annual Report 2024–25**.
    Ask questions and receive AI-generated answers grounded in
    relevant document passages.
    """
)

st.divider()


# --------------------------------------------------
# Cache the embedding model
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return SentenceTransformer(MODEL_NAME)


# --------------------------------------------------
# Save uploaded PDFs safely
# --------------------------------------------------
def save_uploaded_pdf(uploaded_file):
    """Save an uploaded PDF using a content-based filename."""
    file_bytes = uploaded_file.getvalue()

    if len(file_bytes) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise ValueError(
            f"Please upload a PDF smaller than {MAX_UPLOAD_SIZE_MB} MB."
        )

    if not file_bytes.startswith(b"%PDF-"):
        raise ValueError(
            "The uploaded file does not appear to be a valid PDF."
        )

    file_hash = hashlib.sha256(file_bytes).hexdigest()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    destination = UPLOAD_DIR / f"{file_hash}.pdf"

    if not destination.exists():
        destination.write_bytes(file_bytes)

    return destination


# --------------------------------------------------
# Select the source document
# --------------------------------------------------
with st.sidebar:
    st.header("📁 Select Document")

    document_choice = st.radio(
        "Choose a document source",
        ["Default ONGC report", "Upload your own PDF"],
        key="document_choice",
    )

    uploaded_file = None

    if document_choice == "Upload your own PDF":
        uploaded_file = st.file_uploader(
            "Upload a PDF document",
            type=["pdf"],
            help=f"Maximum file size: {MAX_UPLOAD_SIZE_MB} MB.",
            key="pdf_uploader",
        )

    st.divider()

    st.header("ℹ️ About DocuQuery")
    st.write("**Embeddings:** Sentence Transformers")
    st.write("**Vector search:** FAISS")
    st.write("**Answer generation:** Google Gemini")
    st.write("**Interface:** Streamlit")


# --------------------------------------------------
# Resolve the selected PDF
# --------------------------------------------------
selected_pdf = None

if document_choice == "Default ONGC report":
    selected_pdf = DEFAULT_PDF_PATH

elif uploaded_file is not None:
    try:
        selected_pdf = save_uploaded_pdf(uploaded_file)
    except (OSError, ValueError) as error:
        st.error(str(error))
        st.stop()
else:
    st.info("Upload a PDF in the sidebar to begin.")
    st.stop()

if not selected_pdf.exists():
    st.error(f"Document not found: {selected_pdf}")
    st.stop()


# --------------------------------------------------
# Initialize the selected document and its index
# --------------------------------------------------
try:
    with st.status(
        "Preparing your document...",
        expanded=True,
    ) as status:
        st.write("Step 1/2: Loading the embedding model...")
        model = get_embedding_model()

        st.write("Step 2/2: Loading or building the document index...")
        chunks, index = load_or_build_index(
            selected_pdf,
            MODEL_NAME,
            model,
        )

        status.update(
            label="Document is ready!",
            state="complete",
            expanded=False,
        )

except Exception as error:
    st.error("DocuQuery could not prepare the selected PDF.")
    with st.expander("Technical error details"):
        st.exception(error)
    st.stop()


# --------------------------------------------------
# Document metadata
# --------------------------------------------------
try:
    import pymupdf

    with pymupdf.open(selected_pdf) as document:
        page_count = len(document)

except Exception as error:
    st.error("Could not read the selected PDF's page count.")
    with st.expander("Technical error details"):
        st.exception(error)
    st.stop()

document_hash = hashlib.sha256(
    str(selected_pdf.resolve()).encode("utf-8")
).hexdigest()

# Reset history when the selected document changes.
if st.session_state.get("active_document_hash") != document_hash:
    st.session_state.active_document_hash = document_hash
    st.session_state.chat_history = []

st.markdown("### 📄 Document Overview")
st.caption(f"Selected document: {selected_pdf.name}")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Document pages", page_count)

with col2:
    st.metric("Searchable chunks", f"{len(chunks):,}")

with col3:
    st.metric("Passages retrieved per question", "3")

st.divider()


# --------------------------------------------------
# Conversation history
# --------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# --------------------------------------------------
# Question input
# --------------------------------------------------
st.markdown("### 💬 Ask your document")

with st.form("question_form"):
    question = st.text_area(
        "Enter your question",
        placeholder="Ask a question about the selected PDF...",
        height=100,
        max_chars=2000,
    )

    submitted = st.form_submit_button(
        "🔍 Find Answer",
        type="primary",
        use_container_width=True,
    )


# --------------------------------------------------
# Answer generation and sources
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

            st.session_state.chat_history.append(
                {
                    "question": cleaned_question,
                    "answer": result["answer"],
                    "sources": result.get("sources", []),
                }
            )

            st.markdown("### 📝 Answer")
            st.write(result["answer"])

            st.markdown("### 📚 Source Passages")

            sources = result.get("sources", [])

            if sources:
                for number, source in enumerate(sources, start=1):
                    page_number = source.get("page_number", "Unknown")
                    score = source.get("score")
                    source_text = source.get("text", "")

                    label = f"Source {number} · PDF page {page_number}"

                    if score is not None:
                        label += f" · Similarity {score:.3f}"

                    with st.expander(label):
                        st.write(source_text)

                        if score is not None:
                            st.caption(
                                f"Retrieval similarity: {score:.3f}. "
                                "This is not a confidence percentage."
                            )
            else:
                st.info(
                    "No relevant passages were found in this document."
                )

        except Exception as error:
            st.error("An error occurred while answering your question.")
            with st.expander("Technical error details"):
                st.exception(error)


# --------------------------------------------------
# Conversation history display
# --------------------------------------------------
if st.session_state.chat_history:
    st.divider()

    history_col, button_col = st.columns([4, 1])

    with history_col:
        st.markdown("### 🕘 Previous Questions")

    with button_col:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    previous_questions = st.session_state.chat_history[:-1]

    for item in reversed(previous_questions):
        with st.expander(item["question"]):
            st.markdown("**Answer**")
            st.write(item["answer"])

            for source in item["sources"]:
                page_number = source.get("page_number", "Unknown")
                score = source.get("score")
                label = f"PDF page {page_number}"

                if score is not None:
                    label += f" · Similarity {score:.3f}"

                st.caption(label)
