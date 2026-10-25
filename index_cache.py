
import hashlib
import json
from pathlib import Path

import faiss

from rag_pipeline import load_chunks, build_search_index

CACHE_DIR = Path("storage")
CHUNKS_FILE = CACHE_DIR / "chunks.json"
INDEX_FILE = CACHE_DIR / "faiss.index"
MANIFEST_FILE = CACHE_DIR / "manifest.json"


def get_file_hash(file_path):
    """Calculate a SHA-256 hash of the source PDF."""
    digest = hashlib.sha256()

    with open(file_path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def load_or_build_index(pdf_path, model_name, model):
    """Load a matching saved index or build and save a new one."""
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("Checking document cache...", flush=True)
    pdf_hash = get_file_hash(pdf_path)

    if all(
        path.exists()
        for path in (CHUNKS_FILE, INDEX_FILE, MANIFEST_FILE)
    ):
        try:
            with open(MANIFEST_FILE, "r", encoding="utf-8") as file:
                manifest = json.load(file)

            if (
                manifest.get("pdf_hash") == pdf_hash
                and manifest.get("model_name") == model_name
            ):
                print("Loading saved chunks...", flush=True)

                with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
                    chunks = json.load(file)

                print("Loading saved FAISS index...", flush=True)
                index = faiss.read_index(str(INDEX_FILE))

                if index.ntotal == len(chunks) and chunks:
                    print("Saved index loaded successfully.", flush=True)
                    return chunks, index

                print("Cache is inconsistent. Rebuilding it.", flush=True)

        except (OSError, ValueError, KeyError, RuntimeError) as error:
            print(f"Could not use saved cache: {error}", flush=True)

    print("Building chunks from PDF...", flush=True)
    chunks = load_chunks(str(pdf_path))

    if not chunks:
        raise ValueError("No extractable text found in the PDF.")

    print(f"Created {len(chunks)} chunks.", flush=True)
    print("Generating embeddings and building FAISS index...", flush=True)

    index = build_search_index(chunks, model)

    print("Saving chunks and FAISS index...", flush=True)

    with open(CHUNKS_FILE, "w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False)

    faiss.write_index(index, str(INDEX_FILE))

    with open(MANIFEST_FILE, "w", encoding="utf-8") as file:
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
