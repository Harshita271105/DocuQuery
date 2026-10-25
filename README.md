# DocuQuery — AI-Powered Document Question Answering

**Ask questions about your documents and get AI-generated answers grounded in relevant text passages.**

DocuQuery is a Retrieval-Augmented Generation (RAG) application that lets users ask questions about a PDF document through a web interface. It retrieves relevant passages using semantic search and uses Google's Gemini API to generate answers based on the retrieved context.

The project currently uses the **ONGC Annual Report 2024–25** as its source document.

## ✨ Features

- **PDF text extraction:** Extracts text from PDF pages using PyMuPDF.
- **Text chunking:** Splits extracted text into overlapping chunks to preserve context across boundaries.
- **Semantic search:** Uses Sentence Transformers to convert text and questions into vector embeddings.
- **Vector similarity search:** Uses FAISS to retrieve the most relevant passages for a question.
- **AI-powered answers:** Uses the Gemini API to generate answers grounded in retrieved document passages.
- **Source page references:** Includes page numbers associated with retrieved passages.
- **Persistent indexing:** Saves processed chunks and the FAISS index so they can be reused instead of rebuilt unnecessarily.
- **Interactive web interface:** Provides a Streamlit interface for asking questions.
- **Environment-based configuration:** Loads the Gemini API key from a `.env` file rather than hardcoding credentials.

## 🧰 Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application and RAG pipeline |
| Streamlit | Interactive web interface |
| PyMuPDF | PDF text extraction |
| Sentence Transformers | Text embedding generation |
| `all-MiniLM-L6-v2` | Embedding model |
| FAISS | Efficient vector similarity search |
| Google Gemini API | Context-based answer generation |
| SQLite | Not currently required by the RAG pipeline |
| python-dotenv | Environment variable management |
| Git and GitHub | Version control and source code hosting |

## 🏗️ How It Works

DocuQuery follows a Retrieval-Augmented Generation workflow.

1. **Document ingestion:** The application reads the PDF and extracts its text.
2. **Chunking:** Text is split into chunks of 1,000 characters with an overlap of 200 characters.
3. **Embedding generation:** The embedding model converts each chunk into a numerical vector.
4. **Indexing:** FAISS stores the vectors for efficient similarity search.
5. **Question embedding:** When a user asks a question, the same embedding model converts it into a vector.
6. **Retrieval:** FAISS retrieves the three most similar chunks.
7. **Answer generation:** The retrieved passages and question are sent to Gemini with instructions to answer using the supplied document context.
8. **Source presentation:** The application displays the generated answer and associated source information.

### Architecture

```text
PDF Document
     |
     v
Text Extraction (PyMuPDF)
     |
     v
Text Chunking
(1000 characters, 200 overlap)
     |
     v
Embeddings (Sentence Transformers)
     |
     v
FAISS Vector Index
     |
     v
User Question
     |
     v
Question Embedding
     |
     v
Top-3 Relevant Chunks
     |
     v
Gemini API
     |
     v
Answer + Source Information
     |
     v
Streamlit Interface
```

## 📁 Project Structure

```text
DocuQuery/
├── app.py                 # Streamlit web application
├── rag_pipeline.py        # Retrieval and answer-generation pipeline
├── extract_pdf_text.py    # PDF text extraction utility
├── inspect_pdf.py         # PDF inspection utility
├── chunk_text.py          # Text chunking utility
├── vector_search.py       # Vector search functionality
├── index_cache.py         # Persistent chunk and index caching
├── test_app.py            # Application-related tests
├── test_gemini.py         # Gemini API connection test
├── requirements.txt       # Python dependencies
├── .gitignore             # Files excluded from Git
├── .env                   # Local API credentials (not committed)
├── data/                  # Local source documents (not committed)
└── storage/               # Local cached chunks and FAISS index (not committed)
```

## ⚙️ Getting Started

### Prerequisites

- Python 3.11 or later recommended
- Git
- A Google Gemini API key
- Internet access for installing dependencies and calling the Gemini API

The project was developed and tested using Python 3.13.

### 1. Clone the repository

```bash
git clone https://github.com/Harshita271105/DocuQuery.git
cd DocuQuery
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks environment activation, follow the Python virtual-environment instructions appropriate for your system.

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Replace the placeholder with your own API key. Never commit `.env` or expose your key in screenshots, logs, or public repositories.

### 5. Add the source PDF

Place your permitted PDF document in the `data/` directory.

The current application is configured to use:

```text
data/ar2024-25.pdf
```

If you use a different PDF, update the configured PDF path in the application code accordingly.

### 6. Run the application

```bash
streamlit run app.py
```

Open the local URL printed in the terminal, usually:

```text
http://localhost:8501
```

Enter a question about the document and review the generated answer and source information.

## 🧪 Testing

The Gemini connection can be checked independently:

```bash
python test_gemini.py
```

The command-line RAG pipeline can be tested with:

```bash
python rag_pipeline.py
```

Ask document-related questions and inspect the answers and source page information. The Streamlit application can then be tested separately through the browser.

## ⚡ Performance and Caching

Generating embeddings for a large document can be computationally expensive. DocuQuery uses a persistent cache for processed chunks and the FAISS index.

When the cached document and model information remain valid, the application can reuse the saved index rather than repeating the entire indexing process.

Initial setup may take longer because the embedding model and dependencies need to be downloaded or initialized.

## 🔒 Security and Limitations

- API credentials must remain in local environment variables and must not be committed to Git.
- The source PDF and generated cache files are intended to remain local unless there is a deliberate reason to share them.
- Answers depend on the quality of PDF text extraction and the relevance of retrieved passages.
- Scanned PDFs may require OCR if their pages do not contain extractable text.
- A language model can still produce incorrect or incomplete answers. Verify important claims against the cited source pages.
- Gemini API availability, rate limits, quotas, and usage charges depend on the account and model.
- The current implementation retrieves the top three passages. More advanced retrieval, reranking, and evaluation can be added in future iterations.

## 🚀 Future Improvements

- Improve source citations with clearer passage previews and page references.
- Add retrieval-quality evaluation and more comprehensive automated tests.
- Support uploading PDFs through the web interface.
- Add conversation history and follow-up questions.
- Improve handling of scanned PDFs and tables.
- Add more robust error handling, retries, and API failure messages.
- Evaluate chunk sizes, retrieval settings, and embedding models.
- Explore deployment options and configuration for a production environment.

## 🎯 Learning Outcomes

This project provides practical experience with:

- Retrieval-Augmented Generation (RAG)
- Natural language processing and semantic embeddings
- Vector databases and similarity search concepts
- PDF parsing and text preprocessing
- Integration with a generative AI API
- Building interactive Python applications
- Caching, debugging, and version control

## 👩‍💻 Author

**Harshita Verma**

Electronics & Computer Engineering student interested in AI/ML, intelligent systems, and software development.

**GitHub:** [Harshita271105](https://github.com/Harshita271105)

---

*DocuQuery is an educational project exploring document-grounded question answering with retrieval and generative AI.*
