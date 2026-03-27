import pdfplumber
import glob

# Find the PDF file
pdf_files = glob.glob(r"D:\11PrabeshX\Projects\latest\Topo-Brain\docs\*.pdf")
if pdf_files:
    pdf_path = pdf_files[0]
    print(f"Reading: {pdf_path}")
    
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for i, page in enumerate(pdf.pages):
            text += f"\n--- Page {i+1} ---\n"
            text += page.extract_text()
            text += "\n"
    
    # Save to file
    with open("blueprint_extracted.txt", "w", encoding="utf-8") as f:
        f.write(text)
    
    print("\nExtraction complete!")
else:
    print("No PDF found")
