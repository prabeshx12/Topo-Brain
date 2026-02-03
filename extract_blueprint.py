"""Extract text from the research blueprint PDF."""
import sys

try:
    import PyPDF2
    pdf_path = "docs/Research Blueprint_ Topology-Preserving 3T-to-7T MRI Enhancement for Early Alzheimer's Detection.pdf"
    
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n\n"
    
    print(text)
except ImportError:
    print("PyPDF2 not available. Trying pdfplumber...")
    try:
        import pdfplumber
        pdf_path = "docs/Research Blueprint_ Topology-Preserving 3T-to-7T MRI Enhancement for Early Alzheimer's Detection.pdf"
        
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n\n"
        
        print(text)
    except ImportError:
        print("Neither PyPDF2 nor pdfplumber available.")
        print("Please install one: pip install PyPDF2 or pip install pdfplumber")
        sys.exit(1)
