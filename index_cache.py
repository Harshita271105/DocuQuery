
import hashlib
import json
import re
from pathlib import Path

import faiss

from rag_pipeline import load_chunks, build_search_index


CACHE_DIR = Path("storage")
DEFAULT_PDF_PATH = Path("data/ar2024-25.pdf")


def get_file_hash(file_path):
    """Calculate a SHA-256 hash of a source PDF."""
    digest = hashlib.sha256()

    with open(file_path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def get_cache_dir(pdf_path, pdf_hash, model_name):
    """Choose a separate cache directory for each document and model."""
    pdf_path = Path(pdf_path).resolve()
    default_pdf = DEFAULT_PDF_PATH.resolve()

    # Preserve the existing cache for the original ONGC document.
    if pdf_path == default_pdf:
        return CACHE_DIR

    safe_model_name = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)

    return CACHE_DIR / f"{pdf_hash[:16]}_{safe_model_name}"


def load_or_build_index(pdf_path, model_name, model):
    """Load a matching document cache or build and save a new index."""
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pdf_hash = get_file_hash(pdf_path)
    document_cache = get_cache_dir(pdf_path, pdf_hash, model_name)

    chunks_file = document_cache / "chunks.json"
    index_file = document_cache / "faiss.index"
    manifest_file = document_cache / "manifest.json"

    document_cache.mkdir(parents=True, exist_ok=True)

    print("Checking document cache...", flush=True)

    if all(
        path.exists()
        for path in (chunks_file, index_file, manifest_file)
    ):
        try:
            with open(manifest_file, "r", encoding="utf-8") as file:
                manifest = json.load(file)

            if (
                manifest.get("pdf_hash") == pdf_hash
                and manifest.get("model_name") == model_name
            ):
                print("Loading saved chunks...", flush=True)

                with open(chunks_file, "r", encoding="utf-8") as file:
                    chunks = json.load(file)

                print("Loading saved FAISS index...", flush=True)
                index = faiss.read_index(str(index_file))

                if (
                    chunks
                    and index.ntotal == len(chunks)
                    and manifest.get("chunk_count") == len(chunks)
                ):
                    print("Saved index loaded successfully.", flush=True)
                    return chunks, index

                print("Cache is inconsistent. Rebuilding it.", flush=True)

        except (OSError, ValueError, KeyError, RuntimeError) as error:
            print(f"Could not use saved cache: {error}", flush=True)

    print("Building chunks from PDF...", flush=True)
    chunks = load_chunks(str(pdf_path))

    if not chunks:
        raise ValueError(
            "No extractable text found in the PDF. "
            "The document may be scanned and require OCR."
        )

    print(f"Created {len(chunks)} chunks.", flush=True)
    print("Generating embeddings and building FAISS index...", flush=True)

    index = build_search_index(chunks, model)

    print("Saving chunks and FAISS index...", flush=True)

    with open(chunks_file, "w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False)

    faiss.write_index(index, str(index_file))

    with open(manifest_file, "w", encoding="utf-8") as file:
        json.dump(
            {
                "pdf_hash": pdf_hash,
                "model_name": model_name,
                "chunk_count": len(chunks),
            },
            file,
            indent=2,
        )

    print("Cache saved successfully.", flush=True)

    return chunks, index
