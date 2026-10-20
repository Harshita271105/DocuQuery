
import pymupdf

pdf_path = "data/ar2024-25.pdf"

document = pymupdf.open(pdf_path)

print("PDF opened successfully!")
print("Total pages:", len(document))

page = document[1]

text = page.get_text()
print("Extracted characters:", len(text))

print("\nFirst page text:")
print(text[:1500])

document.close()
