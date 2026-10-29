from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from xml.sax.saxutils import escape
from datetime import datetime
import hashlib
from pathlib import Path
import html
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


# --------------------------------------------------
# Custom UI styling
# --------------------------------------------------
st.markdown(
    """
    <style>
    /* Main content spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Main title */
    h1 {
        font-weight: 750;
        letter-spacing: -1px;
    }

    /* Section headings */
    h2, h3 {
        letter-spacing: -0.3px;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: var(--secondary-background-color);
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }

    /* Buttons */
    .stButton > button,
    .stFormSubmitButton > button {
        border-radius: 9px;
        font-weight: 600;
        min-height: 42px;
        transition: border-color 0.2s ease;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        border-radius: 10px;
        border-color: rgba(128, 128, 128, 0.25);
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border-radius: 10px;
    }

    /* Text input areas */
    .stTextArea textarea {
        border-radius: 9px;
        line-height: 1.6;
    }

    /* Sidebar spacing */
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    /* Subtle horizontal dividers */
    hr {
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
        opacity: 0.45;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_PDF_PATH = Path(PDF_PATH)
UPLOAD_DIR = Path("storage") / "uploads"
MAX_UPLOAD_SIZE_MB = 50


def create_pdf_report(question, answer, sources, document_name):
    """Generate a downloadable PDF containing an answer and its sources."""
    buffer = BytesIO()
    generated_at = datetime.now().strftime("%d %B %Y, %I:%M %p")
    pdf = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title="DocuQuery Answer Report",
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            textColor=colors.HexColor("#2457A7"),
            spaceAfter=16,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SourceText",
            parent=styles["BodyText"],
            fontSize=9,
            leading=13,
            spaceAfter=8,
        )
    )

    story = [
        Paragraph("DocuQuery Answer Report", styles["ReportTitle"]),
        Paragraph(f"<b>Document:</b> {escape(document_name)}", styles["BodyText"]),
        Spacer(1, 8),
        Paragraph("Question", styles["Heading2"]),
        Paragraph(escape(question), styles["BodyText"]),
        Spacer(1, 10),
        Paragraph("Answer", styles["Heading2"]),
        Paragraph(escape(answer).replace("\n", "<br/>"), styles["BodyText"]),
        Spacer(1, 12),
        Paragraph("Supporting Sources", styles["Heading2"]),
        Spacer(1, 6),
        Paragraph(f"<b>Generated:</b> {escape(generated_at)}",styles["BodyText"],),
        Spacer(1, 8),
    ]
    
    if sources:
        for number, source in enumerate(sources, start=1):
            page_number = source.get("page_number", "Unknown")
            score = source.get("score")
            source_text = source.get("text", "")

            story.append(
                Paragraph(
                    f"<b>Source {number} — PDF page {escape(str(page_number))}</b>",
                    styles["Heading3"],
                )
            )

            if score is not None:
                story.append(
                    Paragraph(
                        f"Retrieval similarity: {float(score):.3f}",
                        styles["SourceText"],
                    )
                )

            story.append(
                Paragraph(
                    escape(str(source_text)).replace("\n", "<br/>"),
                    styles["SourceText"],
                )
            )
            story.append(Spacer(1, 8))
    else:
        story.append(
            Paragraph("No supporting sources were returned.", styles["BodyText"])
        )

    pdf.build(story)
    return buffer.getvalue()


# --------------------------------------------------
# Header
# --------------------------------------------------
header_col, badge_col = st.columns([4, 1])

with header_col:
    st.title("📄 DocuQuery")
    st.markdown(
        "### Ask questions. Find answers. Verify every source."
    )

with badge_col:
    st.write("")
    st.write("")
    st.success("AI-Powered RAG")

st.markdown(
    """
    Upload your own PDF or explore the **ONGC Annual Report 2024–25**.
    DocuQuery retrieves relevant passages and generates answers grounded
    in your selected document.
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
# Sidebar: document selection and app information
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 📂 DocuQuery")
    st.caption("Your document research assistant")

    st.divider()

    st.markdown("### Select your document")

    document_choice = st.radio(
        "Document source",
        ["Default ONGC report", "Upload your own PDF"],
        key="document_choice",
        help="Choose the report to search and ask questions about.",
    )

    uploaded_file = None

    if document_choice == "Upload your own PDF":
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help=f"Maximum file size: {MAX_UPLOAD_SIZE_MB} MB.",
            key="pdf_uploader",
        )

    st.divider()

    st.markdown("### ⚙️ How DocuQuery works")

    with st.expander("Technology stack"):
        st.markdown(
            """
            - **Embeddings:** Sentence Transformers
            - **Vector search:** FAISS
            - **Answer generation:** Google Gemini
            - **PDF processing:** PyMuPDF
            - **Interface:** Streamlit
            """
        )

    st.divider()
    st.caption("AI-powered document question answering")


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
        from PIL import Image
        import io

        with pymupdf.open(selected_pdf) as pdf_document:
            page = pdf_document[int(preview_page) - 1]

            # Render the original PDF page as an image.
            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(1.5, 1.5),
                alpha=False,
            )

            page_image = Image.open(
                io.BytesIO(pixmap.tobytes("png"))
            )

        st.caption(f"Page {int(preview_page)} of {page_count}")

        st.image(
            page_image,
            caption=f"{selected_pdf.name} — Page {int(preview_page)}",
            width=700,
        )

    except Exception as error:
        st.error(f"Could not preview this page: {error}")



# --------------------------------------------------
# Question input
# --------------------------------------------------
st.markdown("## 💬 Ask your document")
st.caption(
    "Ask a specific question about the selected PDF. "
    "Answers are generated from relevant document passages."
)

st.markdown("**Need an idea? Try asking:**")

suggestion_col1, suggestion_col2 = st.columns(2)

with suggestion_col1:
    if st.button(
        "📊 Summarize key findings",
        use_container_width=True,
    ):
        st.session_state["suggested_question"] = (
            "Summarize the key findings of this document."
        )

with suggestion_col2:
    if st.button(
        "📌 Identify important figures",
        use_container_width=True,
    ):
        st.session_state["suggested_question"] = (
            "What are the most important figures and statistics in this document?"
        )

with st.form("question_form"):
    question = st.text_area(
        "Enter your question",
        value=st.session_state.get("suggested_question", ""),
        placeholder="e.g. What were the major financial results?",
        height=120,
        max_chars=2000,
    )

    submitted = st.form_submit_button(
        "🔍 Find Answer",
        type="primary",
        use_container_width=True,
    )

if submitted:
    st.session_state["suggested_question"] = ""


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
                # Use only previous turns, not the current question.
                previous_history = st.session_state.chat_history[-4:]

                result = answer_question(
                    cleaned_question,
                    chunks,
                    model,
                    index,
                    chat_history=previous_history,
                )

            # Keep the latest answer tied to the selected document.
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
# Display the latest answer and source passages
# --------------------------------------------------
latest_result = st.session_state.get(latest_result_key)

if latest_result:
    st.divider()
    st.markdown("## 💬 Answer")

    with st.container(border=True):
        st.markdown("### 📝 Response")
        st.write(latest_result["answer"])

    sources = latest_result.get("sources", [])

    
    # Export the answer and supporting sources
    export_lines = [
        "DocuQuery - Answer Report",
        "=" * 35,
        "",
        f"Document: {selected_pdf.name}",
        f"Question: {latest_result['question']}",
        "",
        "ANSWER",
        "-" * 35,
        latest_result["answer"],
        "",
        "SUPPORTING SOURCES",
        "-" * 35,
    ]

    for number, source in enumerate(sources, start=1):
        page_number = source.get("page_number", "Unknown")
        score = source.get("score")
        source_text = source.get("text", "")

        export_lines.extend([
            "",
            f"Source {number} - PDF page {page_number}",
        ])

        if score is not None:
            export_lines.append(
                f"Retrieval similarity: {score:.3f}"
            )

        export_lines.append(source_text)

    report_text = "\n".join(export_lines)

    st.download_button(
        label="📥 Download answer report (.txt)",
        data=report_text,
        file_name="docuquery_answer_report.txt",
        mime="text/plain",
        key=f"export_answer_{document_hash}",
    )
    st.caption(
    f"Your report includes {len(sources)} supporting source passage(s)."
)
    report = create_pdf_report(
        question=latest_result["question"],
        answer=latest_result["answer"],
        sources=sources,
        document_name=selected_pdf.name,
    )

    st.download_button(
        label="📄 Download answer report (.pdf)",
        data=report,
        file_name="docuquery_answer_report.pdf",
        mime="application/pdf",
        key=f"export_pdf_{document_hash}",
    )


    st.markdown("## 📚 Supporting Evidence")
    st.caption(
        f"{len(sources)} source passage(s) retrieved for this answer."
    )

    if sources:
        for number, source in enumerate(sources, start=1):
            page_number = source.get("page_number", "Unknown")
            score = source.get("score")
            source_text = source.get("text", "")

            title = f"Source {number} · Page {page_number}"
            if score is not None:
                title += f" · Similarity {score:.3f}"

            with st.container(border=True):
                source_col, action_col = st.columns([4, 1])

                with source_col:
                    st.markdown(f"**{title}**")

                try:
                    source_page = int(page_number)
                except (TypeError, ValueError):
                    source_page = None

                with action_col:
                    if (
                        source_page is not None
                        and 1 <= source_page <= page_count
                    ):
                        st.button(
                            "View page",
                            key=f"source_page_{document_hash}_{number}",
                            on_click=navigate_to_source_page,
                            args=(
                                source_page,
                                preview_page_key,
                                open_preview_key,
                            ),
                            use_container_width=True,
                        )

                safe_source_text = html.escape(str(source_text))
                st.markdown(
                    f"""
                    <div style="
                        font-size: 14px;
                        font-weight: 400;
                        line-height: 1.7;
                        overflow-wrap: anywhere;
                        white-space: pre-wrap;
                    ">{safe_source_text}</div>
                    """,
                    unsafe_allow_html=True,
                )

                if score is not None:
                    st.caption(
                        f"Retrieval similarity: {score:.3f} "
                        "(not a confidence percentage)"
                    )
    else:
        st.info("No relevant passages were found in this document.")

# --------------------------------------------------
# Conversation history display
# --------------------------------------------------
if st.session_state.chat_history:
    st.divider()

    history_col, button_col = st.columns([4, 1])

    with history_col:
        st.markdown("## 🕘 Conversation History")
        st.caption("Revisit your previous questions and answers.")

    with button_col:
        st.write("")
        if st.button(
            "🗑️ Clear history",
            use_container_width=True,
        ):
            st.session_state.chat_history = []
            st.session_state.pop(latest_result_key, None)
            st.rerun()

    previous_questions = st.session_state.chat_history[:-1]

    if not previous_questions:
        st.info(
            "Your earlier questions will appear here "
            "after you ask another question."
        )

    for number, item in enumerate(
        reversed(previous_questions),
        start=1,
    ):
        with st.expander(
            f"💬 {item['question']}",
            expanded=False,
        ):
            st.markdown("**Answer**")
            st.write(item["answer"])

            sources = item.get("sources", [])

            if sources:
                st.markdown("**Sources**")

                for source in sources:
                    page_number = source.get(
                        "page_number",
                        "Unknown",
                    )
                    score = source.get("score")

                    source_label = f"PDF page {page_number}"

                    if score is not None:
                        source_label += (
                            f" · Similarity {score:.3f}"
                        )

                    st.caption(f"📄 {source_label}")

