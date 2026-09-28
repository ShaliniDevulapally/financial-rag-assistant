from pypdf import PdfReader

pdf_path = "jpmorgan_2025_annual_report.pdf"

reader = PdfReader(pdf_path)

print(f"Total pages: {len(reader.pages)}")

text = ""

for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text + "\n"

print(f"Characters extracted: {len(text):,}")

with open("annual_report.txt", "w", encoding="utf-8") as file:
    file.write(text)

print("PDF extraction complete!")