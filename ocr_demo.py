import streamlit as st
import pytesseract
from pdf2image import convert_from_bytes
from pypdf import PdfReader
from io import BytesIO

st.set_page_config(
    page_title="Insurance OCR Demo",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Insurance Document OCR")
st.write(
    "Upload a scanned insurance PDF to extract text using Tesseract OCR."
)

uploaded_file = st.file_uploader(
    "Upload a scanned PDF",
    type=["pdf"]
)

if uploaded_file:
    pdf_bytes = uploaded_file.getvalue()

    if st.button("Extract Text"):
        with st.spinner("Extracting text from the PDF..."):
            try:
                # First, check whether ordinary PDF text extraction works
                reader = PdfReader(BytesIO(pdf_bytes))

                extracted_pages = []
                ocr_pages = []

                for page_number, page in enumerate(reader.pages, start=1):
                    text = (page.extract_text() or "").strip()

                    if len(text) > 30:
                        extracted_pages.append(
                            f"--- Page {page_number} (PDF text) ---\n{text}"
                        )
                    else:
                        ocr_pages.append(page_number)

                # Use OCR only for pages with little or no extractable text
                if ocr_pages:
                    images = convert_from_bytes(
                        pdf_bytes,
                        dpi=200,
                        first_page=min(ocr_pages),
                        last_page=max(ocr_pages)
                    )

                    for page_number in ocr_pages:
                        image = images[page_number - min(ocr_pages)]
                        text = pytesseract.image_to_string(
                            image,
                            lang="eng"
                        )

                        extracted_pages.append(
                            f"--- Page {page_number} (Tesseract OCR) ---\n{text}"
                        )

                result = "\n\n".join(extracted_pages)

                st.success("Text extraction completed!")
                st.text_area(
                    "Extracted Text",
                    result,
                    height=400
                )

                st.download_button(
                    "Download Extracted Text",
                    data=result,
                    file_name="extracted_insurance_text.txt",
                    mime="text/plain"
                )

                st.info(
                    f"Pages in PDF: {len(reader.pages)} | "
                    f"Pages processed with OCR: {len(ocr_pages)}"
                )

            except Exception as e:
                st.error(f"Extraction failed: {e}")