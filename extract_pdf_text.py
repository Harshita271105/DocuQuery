import pymupdf
from chunk_text import chunk_text

pdf_path = "data/ar2024-25.pdf"

document = pymupdf.open(pdf_path)

pages = []

print("PDF opened successfully!")
print("Total pages:", len(document))


for page_number, page in enumerate(document, start=1):
    text = page.get_text()

    page_data = {
        "page_number": page_number,
        "text": text
    }

    pages.append(page_data)

print("Pages processed:", len(pages))
print("Characters on page 2:", len(pages[1]["text"]))


empty_pages = 0

for page_data in pages:
    if not page_data["text"].strip():
        empty_pages += 1

print("Pages with no extractable text:", empty_pages)

print("Sample text from page 2:")
print(pages[1]["text"][:300])


all_chunks = []

for page_data in pages:
    page_chunks = chunk_text(page_data["text"])

    for chunk in page_chunks:
        all_chunks.append({
            "page_number": page_data["page_number"],
            "text": chunk
        })

print("Total chunks:", len(all_chunks))
print("First chunk source page:", all_chunks[0]["page_number"])
print("First chunk preview:")
print(all_chunks[0]["text"][:300])



document.close()
