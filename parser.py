import pdfplumber
import io
import os
import tempfile
from pathlib import Path
from PIL import Image

rxlens_ocr = None
rxlens_normalizer = None
RXLENS_LOADED = False

def load_rxlens():
    global rxlens_ocr, rxlens_normalizer, RXLENS_LOADED
    if RXLENS_LOADED:
        return rxlens_ocr
    RXLENS_LOADED = True
    try:
        from rxlens_model__2___1_ import OCRPipeline, ImageNormalizer
        rxlens_ocr = OCRPipeline()
        rxlens_normalizer = ImageNormalizer
        print("RxLens ML Model Loaded!")
    except Exception as e:
        print(f"RxLens fallback: {e}")
        rxlens_ocr = None
    return rxlens_ocr

def extract_text(file_bytes: bytes, filename: str) -> str:
    try:
        if filename.endswith(".pdf"):
            return extract_from_pdf(file_bytes)
        elif filename.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
            return extract_from_image(file_bytes)
        elif filename.endswith(".txt"):
            return file_bytes.decode("utf-8", errors="ignore")
        else:
            try:
                return extract_from_pdf(file_bytes)
            except:
                return extract_from_image(file_bytes)
    except Exception as e:
        print(f"PARSER ERROR: {e}")
        raise e

def extract_from_pdf(file_bytes: bytes) -> str:
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    result = "\n".join(text_parts)
    if not result.strip():
        result = extract_pdf_with_ocr(file_bytes)
    return result

def extract_pdf_with_ocr(file_bytes: bytes) -> str:
    try:
        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        all_text = []
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_bytes = pix.tobytes("png")
            text = extract_from_image(img_bytes)
            all_text.append(text)
        return "\n".join(all_text)
    except Exception as e:
        print(f"PDF OCR error: {e}")
        return ""

def extract_from_image(file_bytes: bytes) -> str:
    ocr = load_rxlens()
    if ocr is not None:
        try:
            image = Image.open(io.BytesIO(file_bytes))
            if image.mode != "RGB":
                image = image.convert("RGB")
            normalized = rxlens_normalizer.normalize(image)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                normalized.save(tmp.name)
                tmp_path = tmp.name
            text = ocr.extract(tmp_path)
            os.unlink(tmp_path)
            print(f"RxLens extracted {len(text)} chars")
            return text.strip()
        except Exception as e:
            print(f"RxLens image error: {e}")

    # Tesseract fallback
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        image = Image.open(io.BytesIO(file_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
        return pytesseract.image_to_string(image, lang="eng").strip()
    except Exception as e:
        print(f"Tesseract error: {e}")
        return ""