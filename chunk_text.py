
def chunk_text(text, chunk_size=1000, overlap=200):
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "chunk_size must be positive and overlap "
            "must be between 0 and chunk_size - 1."
        )

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        if end >= len(text):
            break

        start += chunk_size - overlap

    return chunks


import pymupdf


pdf_path = "data/ar2024-25.pdf"
all_chunks = []

document = pymupdf.open(pdf_path)

for page_number, page in enumerate(document, start=1):
    text = page.get_text().strip()

    if not text:
        continue

    page_chunks = chunk_text(text)

    for chunk_index, chunk in enumerate(page_chunks, start=1):
        all_chunks.append({
            "page_number": page_number,
            "chunk_index": chunk_index,
            "text": chunk
        })

document.close()

print("Total chunks:", len(all_chunks))
print("First chunk page:", all_chunks[0]["page_number"])
print("First chunk length:", len(all_chunks[0]["text"]))
print("First chunk preview:")
print(all_chunks[0]["text"][:300])

