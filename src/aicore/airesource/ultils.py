import fitz
import pdfplumber
import io
from PIL import Image
import re

def format_llm(s):
    return s.split("</think>")[-1]


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def extract_figure_from_pdf(pdf_file, figure_number: int):
    """
    Extracts the image corresponding to 'hình {figure_number}' from the PDF.
    Returns a PIL Image or None.
    """
    with fitz.open(stream=pdf_file.getvalue(), filetype="pdf") as pdf_document:
        count = 0
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                count += 1
                if count == figure_number:
                    xref = img[0]
                    image_info = pdf_document.extract_image(xref)
                    image_bytes = image_info["image"]
                    return Image.open(io.BytesIO(image_bytes))
    return None


def extract_table_from_pdf(pdf_file, table_number: int):
    """
    Extracts the table corresponding to 'bảng {table_number}' from the PDF.
    Returns table text or None.
    """
    count = 0
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                count += 1
                if count == table_number:
                    # Convert table to text
                    table_text = "\n".join(
                        ["\t".join([str(cell) for cell in row]) for row in table])
                    return table_text
    return None


def detect_figure_or_table_query(query: str):
    """
    Detect if the query is asking for a figure or table explanation.
    Returns ('figure', number) or ('table', number) or (None, None)
    Handles various keywords: hình, fig, fig., ảnh, hình ảnh, bảng, table, tbl, tab, etc.
    """
    # Patterns for figure
    figure_patterns = [
        r"Hình\s*(\d+)",
        r"hình\s*(\d+)",
        r"hình\s*ảnh\s*(\d+)",
        r"ảnh\s*(\d+)",
        r"fig(?:\.|ure)?\s*(\d+)",   # fig 1, fig. 1, figure 1
        r"image\s*(\d+)",
        r"h\.?\s*(\d+)",              # h. 1 (rare, but possible)
    ]
    for pat in figure_patterns:
        match_figure = re.search(pat, query, re.IGNORECASE)
        if match_figure:
            return ("Hình", int(match_figure.group(1)))

    # Patterns for table
    table_patterns = [
        r"Bảng\s*(\d+)",
        r"bảng\s*(\d+)",
        r"table\s*(\d+)",
        r"tbl\.?\s*(\d+)",
        r"tab\.?\s*(\d+)",
        r"t\.?\s*(\d+)",              # t. 1 (rare, but possible)
    ]
    for pat in table_patterns:
        match_table = re.search(pat, query, re.IGNORECASE)
        if match_table:
            return ("Bảng", int(match_table.group(1)))

    return (None, None)
