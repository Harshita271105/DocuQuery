
import pymupdf
import faiss
from sentence_transformers import SentenceTransformer


PDF_PATH = "data/ar2024-25.pdf"
CHUNK_SIZE = 1000
OVERLAP = 200
MODEL_NAME = "all-MiniLM-L6-v2"


def load_chunks(pdf_path):
    document = pymupdf.open(pdf_path)
    chunks = []

    try:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text().strip()

            if not text:
                continue

            start = 0

            while start < len(text):
                end = min(start + CHUNK_SIZE, len(text))
                chunk_text = text[start:end]

                chunks.append({
                    "page_number": page_number,
                    "text": chunk_text
                })

                if end >= len(text):
                    break

                start += CHUNK_SIZE - OVERLAP
    finally:
        document.close()

    return chunks


def main():
    print("Loading PDF and creating chunks...")
    chunks = load_chunks(PDF_PATH)
    print("Total chunks:", len(chunks))

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    texts = [chunk["text"] for chunk in chunks]

    print("Generating embeddings...")
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = embeddings.astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    print("FAISS index created.")
    print("Vectors indexed:", index.ntotal)

    while True:
        question = input("\nAsk a question (or type 'exit'): ").strip()

        if question.lower() == "exit":
            break

        if not question:
            print("Please enter a question.")
            continue

        query_embedding = model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        k = min(3, len(chunks))
        scores, indices = index.search(query_embedding, k)

        print("\nMost relevant passages:")

        for rank, (score, chunk_index) in enumerate(
            zip(scores[0], indices[0]), start=1
        ):
            chunk = chunks[int(chunk_index)]

            print(f"\nResult {rank}")
            print("Page:", chunk["page_number"])
            print("Similarity score:", round(float(score), 4))
            print("Text:", chunk["text"][:700])


if __name__ == "__main__":
    main()
