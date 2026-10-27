import hashlib
from pathlib import Path

import pymupdf
import streamlit as st
from sentence_transformers import SentenceTransformer

from index_cache import load_or_build_index
from rag_pipeline import PDF_PATH, MODEL_NAME, answer_question


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
    Ask questions and receive AI-generated answers grounded in relevant
    document passages.
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
# Save uploaded PDFs
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
    with st.status("Preparing your document...", expanded=True) as status:
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
    with pymupdf.open(selected_pdf) as document:
        page_count = len(document)
except Exception as error:
    st.error("Could not read the selected PDF's page count.")
    with st.expander("Technical error details"):
        st.exception(error)
    st.stop()

# Use the file contents, not just its path, to identify the document.
document_hash = hashlib.sha256(selected_pdf.read_bytes()).hexdigest()

if st.session_state.get("active_document_hash") != document_hash:
    st.session_state.active_document_hash = document_hash
    st.session_state.chat_history = []
    st.session_state.pop(f"latest_result_{document_hash}", None)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

preview_page_key = f"preview_page_{document_hash}"
pending_page_key = f"pending_preview_page_{document_hash}"
open_preview_key = f"open_pdf_preview_{document_hash}"
latest_result_key = f"latest_result_{document_hash}"

# Apply a pending page before creating the page-number widget.
# This assignment happens before st.number_input is rendered.
if pending_page_key in st.session_state:
    requested_page = int(st.session_state.pop(pending_page_key))
    st.session_state[preview_page_key] = max(
        1, min(requested_page, page_count)
    )


def navigate_to_source_page(page_number, page_key, open_key):
    """Set the preview page before Streamlit reruns the app."""
    st.session_state[page_key] = int(page_number)
    st.session_state[open_key] = True


# --------------------------------------------------
# Document overview
# --------------------------------------------------
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
# PDF preview and page navigation
# --------------------------------------------------
st.markdown("### 📖 PDF Preview")

with st.expander(
    "View and navigate the original PDF",
    expanded=st.session_state.get(open_preview_key, False),
):
    preview_page = st.number_input(
        "Enter PDF page number",
        min_value=1,
        max_value=max(1, page_count),
        step=1,
        key=preview_page_key,
    )

    try:
        with pymupdf.open(selected_pdf) as pdf_document:
            page = pdf_document[int(preview_page) - 1]
            page_text = page.get_text("text")

        st.caption(f"Page {int(preview_page)} of {page_count}")

        if page_text.strip():
            st.text_area(
                "Page content",
                value=page_text,
                height=350,
                disabled=True,
                key=f"preview_content_{document_hash}_{int(preview_page)}",
            )
        else:
            st.info(
                "No extractable text was found on this page. "
                "It may contain scanned images."
            )
    except Exception as error:
        st.error(f"Could not preview this page: {error}")


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
# Answer generation
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

            result_record = {
                "question": cleaned_question,
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
            }
            st.session_state[latest_result_key] = result_record
            st.session_state.chat_history.append(result_record)

        except Exception as error:
            st.error("An error occurred while answering your question.")
            with st.expander("Technical error details"):
                st.exception(error)


# --------------------------------------------------
# Display the latest answer and source passages on every rerun
# This keeps source buttons available after clicking one.
# --------------------------------------------------
latest_result = st.session_state.get(latest_result_key)

if latest_result:
    st.markdown("### 📝 Answer")
    st.write(latest_result["answer"])

    st.markdown("### 📚 Source Passages")
    sources = latest_result.get("sources", [])

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

                # Accept ordinary Python integers and integer-like values,
                # but reject missing or nonnumeric page numbers.
                try:
                    source_page = int(page_number)
                except (TypeError, ValueError):
                    source_page = None

                if source_page is not None and 1 <= source_page <= page_count:
                    st.button(
                        f"Preview PDF page {source_page}",
                        key=f"source_page_{document_hash}_{number}",
                        on_click=navigate_to_source_page,
                        args=(source_page, preview_page_key, open_preview_key),
                    )
    else:
        st.info("No relevant passages were found in this document.")


# --------------------------------------------------
# Conversation history
# --------------------------------------------------
if st.session_state.chat_history:
    st.divider()
    history_col, button_col = st.columns([4, 1])

    with history_col:
        st.markdown("### 🕘 Previous Questions")

    with button_col:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.pop(latest_result_key, None)
            st.rerun()

    previous_questions = st.session_state.chat_history[:-1]
    for item in reversed(previous_questions):
        with st.expander(item["question"]):
            st.markdown("**Answer**")
            st.write(item["answer"])

            for source in item.get("sources", []):
                page_number = source.get("page_number", "Unknown")
                score = source.get("score")
                label = f"PDF page {page_number}"

                if score is not None:
                    label += f" · Similarity {score:.3f}"

                st.caption(label)
