import os
import fitz  
import pdfplumber
import pandas as pd

# define project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# path to the PDF inside 'data' folder
pdf_path = os.path.join(BASE_DIR, "data", "doc1.pdf")

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF and returns a DataFrame."""
    doc = fitz.open(pdf_path)
    text_data = [{"page": i + 1, "content": page.get_text("text")} for i, page in enumerate(doc)]
    return pd.DataFrame(text_data)

'''
def extract_tables_from_pdf(pdf_path):
    """Extracts tables from a PDF and returns a DataFrame."""
    tables_data = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for table in tables:
                df_table = pd.DataFrame(table)
                df_table["page"] = i + 1
                tables_data.append(df_table)
    return pd.concat(tables_data, ignore_index=True) if tables_data else pd.DataFrame()
'''

