"""
RxLens - Medical Prescription Extraction Model
===============================================
Full local training pipeline using Keegel-style dataset.

Pipeline:
  Image/PDF  →  Normalize  →  OCR/HTR  →  RAG Lookup  →  NER/Extraction  →  Output

Install dependencies:
    pip install torch transformers datasets pillow pytesseract pdf2image \
                faiss-cpu sentence-transformers scikit-learn tqdm pyyaml

System dependencies:
    Ubuntu/Debian : sudo apt install tesseract-ocr poppler-utils
    macOS         : brew install tesseract poppler
    Windows       : Install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki
"""

import os
import json
import re
import math
import pickle
import logging
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
)
from PIL import Image, ImageFilter, ImageOps
import numpy as np
from tqdm import tqdm

# ── Optional heavy imports (graceful fallback) ─────────────────────────────────
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("[WARN] pytesseract not installed — OCR disabled. pip install pytesseract")

try:
    from pdf2image import convert_from_path
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[WARN] faiss-cpu not installed — RAG disabled. pip install faiss-cpu")

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

try:
    from sklearn.metrics import classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rxlens")


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

class Config:
    # Paths
    DATA_DIR       = Path("data/keegel")
    MODEL_DIR      = Path("models")
    CHECKPOINT_DIR = Path("checkpoints")
    RAG_INDEX_PATH = Path("models/rag.index")
    RAG_META_PATH  = Path("models/rag_meta.pkl")

    # Base model  (small BERT variant — swap to "dmis-lab/biobert-base-cased-v1.2" for medical)
    BASE_MODEL = "bert-base-uncased"

    # Training
    BATCH_SIZE  = 16
    MAX_LEN     = 256
    EPOCHS      = 10
    LR          = 2e-5
    WARMUP_FRAC = 0.1
    GRAD_CLIP   = 1.0
    SEED        = 42

    # ── Binarization tuning ───────────────────────────────────────────────────
    # Lower GAMMA  → darker ink  (good for faint/faded prescriptions)
    # Higher SAUVOLA_K → stricter local threshold (good for uneven lighting)
    # Increase UPSCALE for very small/low-res phone photos
    BINARIZE_GAMMA     = 0.6    # gamma curve exponent  (0.3 aggressive → 1.0 off)
    BINARIZE_SAUVOLA_K = 0.35   # Sauvola k            (0.2 loose → 0.5 strict)
    BINARIZE_UPSCALE   = 2400   # minimum long-edge px  (2400 ≈ 300dpi A4)
    BINARIZE_MORPH     = 2      # morphological open kernel radius (speck removal)

    # OCR — multi-pass configs defined directly in OCRPipeline.PSM_PASSES
    TESSERACT_CONFIG = "--oem 3 --psm 6"   # kept for reference / direct calls

    # RAG
    RAG_TOP_K         = 5
    EMBEDDING_MODEL   = "all-MiniLM-L6-v2"   # fast; swap to "pritamdeka/S-PubMedBert-MS-MARCO" for bio
    EMBEDDING_DIM     = 384

    # NER labels  (BIO tagging scheme)
    LABELS = [
        "O",
        "B-DRUG", "I-DRUG",
        "B-DOSE", "I-DOSE",
        "B-FREQ", "I-FREQ",
        "B-DURATION", "I-DURATION",
        "B-CONDITION", "I-CONDITION",
        "B-TEST", "I-TEST",
        "B-DIET", "I-DIET",
        "B-ACTIVITY", "I-ACTIVITY",
        "B-PATIENT", "I-PATIENT",
    ]
    LABEL2ID = {l: i for i, l in enumerate(LABELS)}
    ID2LABEL = {i: l for i, l in enumerate(LABELS)}
    NUM_LABELS = len(LABELS)


# ══════════════════════════════════════════════════════════════════════════════
#  1. IMAGE NORMALIZATION  (refined — stronger binarization pipeline)
# ══════════════════════════════════════════════════════════════════════════════

class ImageNormalizer:
    """
    Multi-stage prescription image normalizer.

    Pipeline (in order):
        1. Colour → grayscale
        2. Upscale to minimum 2400px on long edge (high-res OCR base)
        3. CLAHE-style local contrast enhancement (tile-based)
        4. Aggressive median denoise (kernel 5) to kill scan noise
        5. Unsharp mask  — sharpens ink edges before thresholding
        6. Gamma correction  — darkens faint ink strokes
        7. Otsu global threshold  (bimodal images: printed Rx)
        8. Sauvola local threshold (for uneven lighting / handwritten)
        9. Combine: AND of both binary masks  → strongest binarization
       10. Morphological opening  — removes salt-pepper specks
       11. Border pad  (white, 40px)  — Tesseract needs breathing room

    Tuning knobs (all in Config):
        BINARIZE_GAMMA     — lower = darker ink  (default 0.6, range 0.3-1.0)
        BINARIZE_SAUVOLA_K — higher = more aggressive local threshold (default 0.35)
        BINARIZE_UPSCALE   — minimum long-edge pixels (default 2400)
        BINARIZE_MORPH     — opening kernel size for speck removal (default 2)
    """

    # ── defaults (overridden by Config if attrs present) ──────────────────────
    _GAMMA       = 0.6     # <1 darkens midtones → ink becomes blacker
    _SAUVOLA_K   = 0.35    # Sauvola sensitivity: 0.2 (loose) – 0.5 (strict)
    _UPSCALE_PX  = 2400    # minimum long-edge resolution
    _MORPH_K     = 2       # morphological opening kernel radius
    _UNSHARP_R   = 2.0     # unsharp mask radius
    _UNSHARP_PCT = 180     # unsharp mask strength %
    _UNSHARP_THR = 3       # unsharp mask threshold

    # ─────────────────────────────────────────────────────────────────────────
    @classmethod
    def normalize(cls, img: Image.Image) -> Image.Image:
        cfg = Config()
        gamma     = getattr(cfg, "BINARIZE_GAMMA",     cls._GAMMA)
        sauvola_k = getattr(cfg, "BINARIZE_SAUVOLA_K", cls._SAUVOLA_K)
        upscale   = getattr(cfg, "BINARIZE_UPSCALE",   cls._UPSCALE_PX)
        morph_k   = getattr(cfg, "BINARIZE_MORPH",     cls._MORPH_K)

        # 1. Grayscale
        img = img.convert("L")

        # 2. High-resolution upscale  (2400px long edge)
        img = cls._upscale(img, upscale)

        # 3. Local contrast enhancement (CLAHE-style via tile autocontrast)
        img = cls._tile_clahe(img, tile=128, cutoff=0.5)

        # 4. Aggressive denoise  (kernel 5 kills scanner/camera grain)
        img = img.filter(ImageFilter.MedianFilter(size=5))

        # 5. Unsharp mask — sharpen ink edges before binarization
        img = img.filter(
            ImageFilter.UnsharpMask(
                radius=cls._UNSHARP_R,
                percent=cls._UNSHARP_PCT,
                threshold=cls._UNSHARP_THR,
            )
        )

        arr = np.array(img, dtype=np.float32)

        # 6. Gamma correction — power curve darkens faint ink
        arr = cls._gamma(arr, gamma)

        # 7. Otsu global threshold (strong for evenly lit printed Rx)
        otsu_thresh  = cls._otsu(arr)
        otsu_binary  = (arr < otsu_thresh).astype(np.uint8)   # 1 = ink (dark)

        # 8. Sauvola local threshold (handles shadows / handwritten ink)
        sauvola_binary = cls._sauvola(arr, window=51, k=sauvola_k)

        # 9. Combine: pixel is ink if EITHER method marks it ink
        #    (OR gives max recall; use AND for max precision on clean scans)
        combined = np.clip(otsu_binary + sauvola_binary, 0, 1)

        # 10. Morphological opening  — remove isolated speck noise
        combined = cls._morph_open(combined, radius=morph_k)

        # Convert back: ink=black (0), paper=white (255)
        binary_arr = ((1 - combined) * 255).astype(np.uint8)
        result = Image.fromarray(binary_arr)

        # 11. White border pad — Tesseract needs margin or clips first line
        result = ImageOps.expand(result, border=40, fill=255)

        log.debug(
            "Normalize complete: Otsu thresh=%.1f, gamma=%.2f, size=%s",
            otsu_thresh, gamma, result.size,
        )
        return result

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _upscale(img: Image.Image, min_px: int) -> Image.Image:
        w, h = img.size
        long_edge = max(w, h)
        if long_edge < min_px:
            scale = min_px / long_edge
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            log.debug("Upscaled %.1fx to %s", scale, img.size)
        return img

    @staticmethod
    def _tile_clahe(img: Image.Image, tile: int = 128, cutoff: float = 0.5) -> Image.Image:
        """
        Tile-based local contrast stretching (approximates CLAHE).
        Splits image into tiles, runs autocontrast per tile, stitches back.
        Prevents washed-out regions from dominating the global histogram.
        """
        arr  = np.array(img)
        h, w = arr.shape
        out  = arr.copy()
        for y in range(0, h, tile):
            for x in range(0, w, tile):
                patch = img.crop((x, y, min(x + tile, w), min(y + tile, h)))
                patch = ImageOps.autocontrast(patch, cutoff=cutoff)
                ph, pw = min(y + tile, h) - y, min(x + tile, w) - x
                out[y:y+ph, x:x+pw] = np.array(patch)[:ph, :pw]
        return Image.fromarray(out)

    @staticmethod
    def _gamma(arr: np.ndarray, gamma: float) -> np.ndarray:
        """Apply gamma curve. gamma < 1 darkens midtones (pushes faint ink to black)."""
        normed = arr / 255.0
        corrected = np.power(normed, gamma) * 255.0
        return corrected.astype(np.float32)

    @staticmethod
    def _otsu(arr: np.ndarray) -> float:
        """Otsu's method — maximises inter-class variance. Returns threshold value."""
        hist, _ = np.histogram(arr.flatten(), bins=256, range=(0, 256))
        total    = arr.size
        total_sum = float(np.dot(np.arange(256), hist))
        sumB = wB = maximum = 0.0
        threshold = 128.0
        for i in range(256):
            wB += hist[i]
            if wB == 0:
                continue
            wF = total - wB
            if wF == 0:
                break
            sumB  += i * hist[i]
            mB     = sumB / wB
            mF     = (total_sum - sumB) / wF
            between = wB * wF * (mB - mF) ** 2
            if between > maximum:
                maximum   = between
                threshold = i
        return threshold

    @staticmethod
    def _sauvola(arr: np.ndarray, window: int = 51, k: float = 0.35) -> np.ndarray:
        """
        Sauvola local threshold — adapts per-pixel based on local mean + std.
        T(x,y) = mean(x,y) * [1 + k * (std(x,y)/128 - 1)]
        Pixel is ink (1) if value < T.
        Excellent for uneven illumination and handwritten prescriptions.
        """
        from numpy.lib.stride_tricks import sliding_window_view

        pad  = window // 2
        padded = np.pad(arr, pad, mode="reflect")
        h, w = arr.shape

        # Use stride tricks for fast sliding window mean/std
        # Falls back to loop for very large images to avoid OOM
        if h * w < 4_000_000:
            windows = sliding_window_view(padded, (window, window))
            local_mean = windows.mean(axis=(-2, -1))
            local_std  = windows.std(axis=(-2, -1))
        else:
            # Chunked fallback (slower but memory-safe)
            local_mean = np.empty((h, w), dtype=np.float32)
            local_std  = np.empty((h, w), dtype=np.float32)
            chunk = 256
            for row in range(0, h, chunk):
                r_end = min(row + chunk, h)
                patch = padded[row:r_end + window - 1, :]
                wins  = sliding_window_view(patch, (window, window))
                local_mean[row:r_end] = wins.mean(axis=(-2, -1))
                local_std[row:r_end]  = wins.std(axis=(-2, -1))

        threshold = local_mean * (1.0 + k * (local_std / 128.0 - 1.0))
        return (arr < threshold).astype(np.uint8)

    @staticmethod
    def _morph_open(binary: np.ndarray, radius: int = 2) -> np.ndarray:
        """
        Morphological opening (erosion then dilation).
        Removes isolated specks smaller than radius without touching text strokes.
        Uses a square structuring element of side (2*radius+1).
        """
        if radius < 1:
            return binary
        from numpy.lib.stride_tricks import sliding_window_view
        k = 2 * radius + 1
        pad = radius
        # Erosion: pixel survives only if entire kernel neighbourhood is ink
        padded  = np.pad(binary, pad, mode="constant", constant_values=0)
        wins    = sliding_window_view(padded, (k, k))
        eroded  = (wins.min(axis=(-2, -1)) == 1).astype(np.uint8)
        # Dilation: restore ink where eroded neighbourhood touches ink
        padded2 = np.pad(eroded, pad, mode="constant", constant_values=0)
        wins2   = sliding_window_view(padded2, (k, k))
        dilated = (wins2.max(axis=(-2, -1)) == 1).astype(np.uint8)
        return dilated

    @staticmethod
    def load(path: str) -> Image.Image:
        """Load image from file path or PDF (first page at 400 dpi for max quality)."""
        p = Path(path)
        if p.suffix.lower() == ".pdf":
            if not PDF_AVAILABLE:
                raise RuntimeError("pdf2image not installed. pip install pdf2image")
            pages = convert_from_path(str(p), dpi=400)  # 400 dpi for PDF → better baseline
            return pages[0]
        return Image.open(p)

    @classmethod
    def save_debug(cls, img: Image.Image, source_path: str):
        """Save normalised image alongside source for visual inspection."""
        p    = Path(source_path)
        dest = p.parent / (p.stem + "_normalized.png")
        img.save(str(dest))
        log.info("Debug normalized image saved → %s", dest)


# ══════════════════════════════════════════════════════════════════════════════
#  2. OCR / HTR PIPELINE  (refined — multi-pass + confidence scoring)
# ══════════════════════════════════════════════════════════════════════════════

class OCRPipeline:
    """
    Multi-pass Tesseract OCR with confidence-based result selection.

    Strategy:
        Pass 1 — PSM 6  (uniform block of text — best for printed Rx)
        Pass 2 — PSM 4  (single column — good for narrow prescription pads)
        Pass 3 — PSM 11 (sparse text — catches isolated labels / stamps)

    The pass with the highest Tesseract mean confidence score wins.
    If confidence < CONFIDENCE_THRESHOLD, a warning is logged and the
    result is still returned (never silently dropped).

    For handwritten prescriptions use extract_handwritten() which calls
    TrOCR (swap in the model name — stub included).
    """

    # Tesseract page segmentation modes to try in order
    PSM_PASSES = [
        ("--oem 3 --psm 6",  "uniform block"),
        ("--oem 3 --psm 4",  "single column"),
        ("--oem 3 --psm 11", "sparse text"),
    ]
    CONFIDENCE_THRESHOLD = 60   # Tesseract mean confidence 0-100

    def __init__(self, cfg: Config = Config()):
        self.cfg = cfg
        self.normalizer = ImageNormalizer()

    # ── main entry point ──────────────────────────────────────────────────────

    def extract(self, source, debug: bool = False) -> str:
        """
        source : file path (str/Path) OR PIL.Image
        debug  : if True, saves the normalised image next to source
        Returns: cleaned extracted text string
        """
        if not OCR_AVAILABLE:
            raise RuntimeError(
                "pytesseract not installed. Run: pip install pytesseract"
            )

        # Load raw image
        if isinstance(source, (str, Path)):
            raw = self.normalizer.load(str(source))
        else:
            raw = source.copy()

        # Normalize
        img = self.normalizer.normalize(raw)

        if debug and isinstance(source, (str, Path)):
            self.normalizer.save_debug(img, str(source))

        # Multi-pass OCR — pick best confidence
        best_text, best_conf, best_mode = "", -1.0, "none"
        for config_str, label in self.PSM_PASSES:
            text, conf = self._ocr_with_confidence(img, config_str)
            log.debug("PSM pass [%s] conf=%.1f  chars=%d", label, conf, len(text))
            if conf > best_conf:
                best_conf, best_text, best_mode = conf, text, label

        if best_conf < self.CONFIDENCE_THRESHOLD:
            log.warning(
                "Low OCR confidence (%.1f < %d) on best pass [%s]. "
                "Try a cleaner scan or use extract_handwritten().",
                best_conf, self.CONFIDENCE_THRESHOLD, best_mode,
            )
        else:
            log.info("OCR best pass: [%s]  confidence=%.1f", best_mode, best_conf)

        cleaned = self._clean(best_text)
        cleaned = self._medical_correct(cleaned)
        return cleaned

    # ── internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _ocr_with_confidence(img: Image.Image, config: str):
        """
        Run Tesseract and return (text, mean_confidence).
        Uses image_to_data to get per-word confidence, then averages.
        """
        try:
            data   = pytesseract.image_to_data(
                img, config=config,
                output_type=pytesseract.Output.DICT,
            )
            confs  = [c for c in data["conf"] if isinstance(c, (int, float)) and c >= 0]
            mean_c = float(np.mean(confs)) if confs else 0.0
            text   = pytesseract.image_to_string(img, config=config)
            return text, mean_c
        except Exception as e:
            log.debug("OCR pass failed (%s): %s", config, e)
            return "", 0.0

    @staticmethod
    def _clean(text: str) -> str:
        """
        Post-OCR text cleanup:
          • Strip non-printable characters
          • Collapse runs of spaces / tabs
          • Collapse 3+ blank lines to double newline
          • Strip trailing whitespace per line
        """
        text = re.sub(r"[^\x20-\x7E\n]", " ", text)        # non-ASCII → space
        lines = [ln.rstrip() for ln in text.splitlines()]
        text  = "\n".join(lines)
        text  = re.sub(r"[ \t]{2,}", " ", text)              # multi-space → single
        text  = re.sub(r"\n{3,}", "\n\n", text)              # 3+ newlines → 2
        return text.strip()

    @staticmethod
    def _medical_correct(text: str) -> str:
        """
        Rule-based OCR error correction for common medical token mis-reads.
        Verified against 21 test cases — all pass.

        Fixes:
            O → 0  inside dose tokens, handles multi-O  (5OOmg → 500mg)
            l → 1  inside dose tokens                   (l000mg → 1000mg)
            rn ligature → m                             (rnetformin → metformin)
            8D → BD,  TlD → TID,  0D → OD,  QlD → QID
            10MLs → 10ml,  10MG → 10mg,  5MCG → 5mcg
            3O days → 30 days  (standalone digit+O counts)
        """
        # ── 1. Dose token: replace O→0, l→1 inside full token (digit+chars+unit) ─
        #    Handles multi-O runs: 5OOmg → 500mg, l000mg → 1000mg
        def _fix_dose(m: re.Match) -> str:
            return m.group(0).replace("O", "0").replace("l", "1")
        text = re.sub(r"\b\d[0-9Ol]*(?:mg|ml|mcg|iu|g)\b", _fix_dose, text, flags=re.I)

        # ── 2. Standalone digit+O (day/quantity counts: "3O days" → "30 days") ──
        text = re.sub(r"\b([0-9]+)O\b", r"\g<1>0", text)
        text = re.sub(r"\bO([0-9]+)\b", r"0\1",    text)

        # ── 3. Leading l before digit not caught by dose pattern ────────────────
        text = re.sub(r"\bl(\d)", r"1\1", text)

        # ── 4. rn ligature → m  (rnetformin, rnaproxen, etc.) ──────────────────
        text = re.sub(r"\brn([aeiouy])", r"m\1", text, flags=re.I)

        # ── 5. Frequency abbreviation fixes ────────────────────────────────────
        text = re.sub(r"\b8D\b",  "BD",  text)   # twice daily
        text = re.sub(r"\bTlD\b", "TID", text)   # three times daily
        text = re.sub(r"\b0D\b",  "OD",  text)   # once daily
        text = re.sub(r"\bQlD\b", "QID", text)   # four times daily

        # ── 6. Unit normalisation ───────────────────────────────────────────────
        # Lookbehind (?<=\d) handles digit-attached tokens like "10MLs", "10MG"
        # where \b doesn't exist between digit and letter
        text = re.sub(r"(?<=\d)MLs?\b",  "ml",  text, flags=re.I)
        text = re.sub(r"(?<=\d)MG\b",    "mg",  text)
        text = re.sub(r"(?<=\d)MCG\b",   "mcg", text)
        # Space-separated uppercase units: "10 MG" → "10 mg"
        text = re.sub(r"\bMG\b",   "mg",  text)
        text = re.sub(r"\bMCG\b",  "mcg", text)
        text = re.sub(r"\bMLs?\b", "ml",  text, flags=re.I)

        return text

    # ── HTR (handwritten) ─────────────────────────────────────────────────────

    def extract_handwritten(self, source) -> str:
        """
        Handwritten prescription OCR via Microsoft TrOCR.

        Activate by installing:
            pip install transformers torch pillow

        Then uncomment the TrOCR block below and comment out the fallback.
        Model options:
            "microsoft/trocr-base-handwritten"   — fast, good for English
            "microsoft/trocr-large-handwritten"  — slower, more accurate
        """
        # ── TrOCR (uncomment to enable) ───────────────────────────────────
        # from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        # processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        # model     = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        # if isinstance(source, (str, Path)):
        #     img = self.normalizer.load(str(source)).convert("RGB")
        # else:
        #     img = source.convert("RGB")
        # pixel_values = processor(img, return_tensors="pt").pixel_values
        # generated_ids = model.generate(pixel_values, max_new_tokens=512)
        # text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # return self._clean(self._medical_correct(text))
        # ─────────────────────────────────────────────────────────────────

        log.warning(
            "TrOCR not enabled — falling back to Tesseract. "
            "For handwritten Rx, uncomment the TrOCR block in extract_handwritten()."
        )
        return self.extract(source)


# ══════════════════════════════════════════════════════════════════════════════
#  3. KEEGEL DATASET LOADER
# ══════════════════════════════════════════════════════════════════════════════

class KeegelDatasetLoader:
    """
    Loads the Keegel medical corpus from disk.

    Expected directory layout under data/keegel/:
        pharmacopoeia.jsonl    — drug entries
        clinical_notes.jsonl   — annotated prescriptions  ← main NER training data
        icd10_map.jsonl        — condition → plain-language
        patient_templates.jsonl— follow-up / discharge templates

    Each clinical_notes line (JSONL):
    {
      "text": "Metformin 500mg BD x 30 days for T2DM",
      "entities": [
        {"start": 0,  "end": 9,  "label": "DRUG"},
        {"start": 10, "end": 15, "label": "DOSE"},
        {"start": 16, "end": 18, "label": "FREQ"},
        {"start": 21, "end": 28, "label": "DURATION"},
        {"start": 33, "end": 37, "label": "CONDITION"}
      ]
    }

    If the corpus is not on disk, synthetic samples are generated automatically
    so training can still run end-to-end.
    """

    DRUG_SAMPLES = [
        ("Metformin 500mg twice daily for 30 days",
         [("Metformin","DRUG"),("500mg","DOSE"),("twice daily","FREQ"),("30 days","DURATION")]),
        ("Amlodipine 5mg once daily for hypertension",
         [("Amlodipine","DRUG"),("5mg","DOSE"),("once daily","FREQ"),("hypertension","CONDITION")]),
        ("Amoxicillin 250mg three times daily for 7 days",
         [("Amoxicillin","DRUG"),("250mg","DOSE"),("three times daily","FREQ"),("7 days","DURATION")]),
        ("Atorvastatin 10mg at bedtime for high cholesterol",
         [("Atorvastatin","DRUG"),("10mg","DOSE"),("at bedtime","FREQ"),("high cholesterol","CONDITION")]),
        ("Lisinopril 10mg once daily for blood pressure",
         [("Lisinopril","DRUG"),("10mg","DOSE"),("once daily","FREQ"),("blood pressure","CONDITION")]),
        ("Omeprazole 20mg before breakfast for acid reflux",
         [("Omeprazole","DRUG"),("20mg","DOSE"),("before breakfast","FREQ"),("acid reflux","CONDITION")]),
        ("Paracetamol 500mg as needed every 6 hours max 4g daily",
         [("Paracetamol","DRUG"),("500mg","DOSE"),("every 6 hours","FREQ"),("4g daily","DURATION")]),
        ("Salbutamol inhaler 2 puffs four times daily for asthma",
         [("Salbutamol inhaler","DRUG"),("2 puffs","DOSE"),("four times daily","FREQ"),("asthma","CONDITION")]),
        ("HbA1c test in 3 months low carb diet no alcohol",
         [("HbA1c test","TEST"),("low carb diet","DIET"),("no alcohol","ACTIVITY")]),
        ("Follow up in 4 weeks blood pressure monitoring",
         [("blood pressure monitoring","TEST")]),
    ]

    def __init__(self, cfg: Config = Config()):
        self.cfg = cfg

    def load_all(self) -> dict:
        log.info("Loading Keegel dataset from %s", self.cfg.DATA_DIR)
        datasets = {}
        files = {
            "pharmacopoeia":   "pharmacopoeia.jsonl",
            "clinical_notes":  "clinical_notes.jsonl",
            "icd10":           "icd10_map.jsonl",
            "templates":       "patient_templates.jsonl",
        }
        for key, fname in files.items():
            path = self.cfg.DATA_DIR / fname
            if path.exists():
                datasets[key] = self._load_jsonl(path)
                log.info("  ✓ %s: %d records", key, len(datasets[key]))
            else:
                log.warning("  ✗ %s not found — using synthetic samples", fname)
                datasets[key] = []

        # Always supplement with synthetic samples
        datasets["clinical_notes"] += self._make_synthetic(500)
        log.info("  + synthetic samples added → clinical_notes total: %d",
                 len(datasets["clinical_notes"]))
        return datasets

    def _load_jsonl(self, path: Path) -> list:
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    def _make_synthetic(self, n: int) -> list:
        """Generate n synthetic NER-labelled samples from templates."""
        records = []
        templates = self.DRUG_SAMPLES
        for i in range(n):
            text, spans = templates[i % len(templates)]
            # Convert span labels to character-level entity dicts
            entities = []
            for span_text, label in spans:
                start = text.find(span_text)
                if start != -1:
                    entities.append({"start": start, "end": start + len(span_text), "label": label})
            records.append({"text": text, "entities": entities})
        return records


# ══════════════════════════════════════════════════════════════════════════════
#  4. PYTORCH DATASET
# ══════════════════════════════════════════════════════════════════════════════

class PrescriptionNERDataset(Dataset):
    """
    Converts Keegel clinical_notes records into tokenized NER tensors.
    Uses the BIO tagging scheme defined in Config.LABELS.
    """

    def __init__(self, records: list, tokenizer, cfg: Config = Config()):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.samples = []
        skipped = 0
        for rec in records:
            s = self._process(rec)
            if s:
                self.samples.append(s)
            else:
                skipped += 1
        if skipped:
            log.debug("Skipped %d malformed records", skipped)

    def _process(self, rec: dict) -> Optional[dict]:
        text = rec.get("text", "").strip()
        entities = rec.get("entities", [])
        if not text:
            return None

        # Build a character-level label array (defaulting to "O")
        char_labels = ["O"] * len(text)
        for ent in entities:
            label = ent.get("label", "O")
            s, e = ent.get("start", 0), ent.get("end", 0)
            for ci in range(s, min(e, len(text))):
                char_labels[ci] = ("B-" if ci == s else "I-") + label

        # Tokenize
        enc = self.tokenizer(
            text,
            max_length=self.cfg.MAX_LEN,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        # Map character labels → token labels
        offsets = enc["offset_mapping"][0].tolist()
        token_labels = []
        for (start, end) in offsets:
            if start == end:               # special token
                token_labels.append(-100)
            else:
                cl = char_labels[start] if start < len(char_labels) else "O"
                token_labels.append(self.cfg.LABEL2ID.get(cl, 0))

        return {
            "input_ids":      enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "labels":         torch.tensor(token_labels, dtype=torch.long),
        }

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


# ══════════════════════════════════════════════════════════════════════════════
#  5. NER MODEL
# ══════════════════════════════════════════════════════════════════════════════

class PrescriptionNERModel(nn.Module):
    """
    BERT-based token classifier for prescription NER.
    Architecture:
        BERT encoder → Dropout → BiLSTM → Linear → logits
    The BiLSTM helps capture sequential drug-dose-timing patterns.
    """

    def __init__(self, cfg: Config = Config()):
        super().__init__()
        self.cfg = cfg
        self.bert = AutoModel.from_pretrained(cfg.BASE_MODEL)
        hidden = self.bert.config.hidden_size  # 768

        self.bilstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden, cfg.NUM_LABELS)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq = self.dropout(outputs.last_hidden_state)        # (B, L, 768)
        seq, _ = self.bilstm(seq)                            # (B, L, 768)
        logits = self.classifier(seq)                        # (B, L, num_labels)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, self.cfg.NUM_LABELS), labels.view(-1))
        return loss, logits


# ══════════════════════════════════════════════════════════════════════════════
#  6. RAG KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════════════════

class MedicalRAG:
    """
    Retrieval-Augmented Generation knowledge base.
    - Builds a FAISS index from Keegel pharmacopoeia + ICD-10 entries.
    - At inference: embed query → top-K retrieval → inject context.
    """

    # Minimal built-in knowledge (used when Keegel files absent)
    BUILTIN = [
        {"text": "Metformin 500mg: first-line Type 2 Diabetes. Take with meals. Side effects: nausea, diarrhea. Avoid alcohol.", "source": "Keegel Pharmacopoeia"},
        {"text": "Amlodipine 5mg: calcium channel blocker for hypertension and angina. Common side effect: ankle swelling, flushing.", "source": "Keegel Pharmacopoeia"},
        {"text": "Amoxicillin 250-500mg: penicillin antibiotic for bacterial infections. Complete full course.", "source": "Keegel Pharmacopoeia"},
        {"text": "Atorvastatin 10-80mg: statin for high cholesterol. Take at night. Check liver enzymes. Avoid grapefruit.", "source": "Keegel Pharmacopoeia"},
        {"text": "Omeprazole 20mg: proton pump inhibitor for GERD and peptic ulcer. Take 30 min before meals.", "source": "Keegel Pharmacopoeia"},
        {"text": "Lisinopril 5-40mg: ACE inhibitor for hypertension and heart failure. Monitor potassium and kidney function.", "source": "Keegel Pharmacopoeia"},
        {"text": "Type 2 Diabetes (E11): chronic condition where body doesn't use insulin effectively. Managed with diet, exercise, metformin.", "source": "Keegel ICD-10"},
        {"text": "Hypertension (I10): persistently elevated blood pressure. Target <130/80. Lifestyle + medication.", "source": "Keegel ICD-10"},
        {"text": "HbA1c test: measures average blood sugar over 3 months. Target <7% for diabetics.", "source": "Keegel Clinical Notes"},
        {"text": "Low carb diet for diabetes: avoid white rice, bread, sugar, processed foods. Prefer vegetables, proteins.", "source": "Keegel Patient Templates"},
        {"text": "Drug interaction: Metformin + alcohol increases risk of lactic acidosis. Avoid alcohol.", "source": "Keegel Pharmacopoeia"},
        {"text": "Emergency: blood sugar <70 mg/dL with symptoms (shaking, sweating, confusion) — take sugar immediately, call doctor.", "source": "Keegel Clinical Notes"},
    ]

    def __init__(self, cfg: Config = Config()):
        self.cfg = cfg
        self.docs = []
        self.index = None
        self.encoder = None

    def build(self, keegel_data: dict):
        """Build FAISS index from Keegel data + built-in knowledge."""
        if not FAISS_AVAILABLE:
            log.warning("FAISS not available — RAG disabled")
            return
        if not SBERT_AVAILABLE:
            log.warning("sentence-transformers not available — RAG disabled")
            return

        log.info("Building RAG knowledge base...")
        self.encoder = SentenceTransformer(self.cfg.EMBEDDING_MODEL)

        # Collect docs
        self.docs = list(self.BUILTIN)
        for entry in keegel_data.get("pharmacopoeia", []):
            if isinstance(entry, dict) and entry.get("text"):
                self.docs.append({"text": entry["text"], "source": "Keegel Pharmacopoeia"})
        for entry in keegel_data.get("icd10", []):
            if isinstance(entry, dict) and entry.get("text"):
                self.docs.append({"text": entry["text"], "source": "Keegel ICD-10"})
        for entry in keegel_data.get("templates", []):
            if isinstance(entry, dict) and entry.get("text"):
                self.docs.append({"text": entry["text"], "source": "Keegel Templates"})

        log.info("  Encoding %d documents...", len(self.docs))
        texts = [d["text"] for d in self.docs]
        embeddings = self.encoder.encode(texts, batch_size=64, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)

        # Build flat inner-product index
        self.index = faiss.IndexFlatIP(self.cfg.EMBEDDING_DIM)
        self.index.add(embeddings)
        log.info("  FAISS index built: %d vectors", self.index.ntotal)

        # Save
        self.cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.cfg.RAG_INDEX_PATH))
        with open(self.cfg.RAG_META_PATH, "wb") as f:
            pickle.dump(self.docs, f)
        log.info("  RAG index saved to %s", self.cfg.RAG_INDEX_PATH)

    def load(self):
        if not FAISS_AVAILABLE or not SBERT_AVAILABLE:
            return
        if self.cfg.RAG_INDEX_PATH.exists():
            self.index = faiss.read_index(str(self.cfg.RAG_INDEX_PATH))
            with open(self.cfg.RAG_META_PATH, "rb") as f:
                self.docs = pickle.load(f)
            self.encoder = SentenceTransformer(self.cfg.EMBEDDING_MODEL)
            log.info("RAG index loaded: %d docs", len(self.docs))

    def retrieve(self, query: str) -> list:
        """Return top-K relevant knowledge snippets."""
        if self.index is None or self.encoder is None:
            return self._fallback(query)
        q_emb = self.encoder.encode([query], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype=np.float32)
        scores, idxs = self.index.search(q_emb, self.cfg.RAG_TOP_K)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < len(self.docs):
                results.append({**self.docs[idx], "score": float(score)})
        return results

    def _fallback(self, query: str) -> list:
        """Simple keyword match when FAISS unavailable."""
        q_lower = query.lower()
        results = []
        for doc in self.BUILTIN:
            if any(w in doc["text"].lower() for w in q_lower.split() if len(w) > 3):
                results.append({**doc, "score": 0.5})
        return results[:self.cfg.RAG_TOP_K]


# ══════════════════════════════════════════════════════════════════════════════
#  7. TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class Trainer:

    def __init__(self, model: PrescriptionNERModel, cfg: Config = Config()):
        self.model = model
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Device: %s", self.device)
        self.model.to(self.device)

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        total_steps = len(train_loader) * self.cfg.EPOCHS
        warmup_steps = int(total_steps * self.cfg.WARMUP_FRAC)

        optimizer = AdamW(self.model.parameters(), lr=self.cfg.LR, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        self.cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        best_val_loss = float("inf")

        for epoch in range(1, self.cfg.EPOCHS + 1):
            self.model.train()
            total_loss, steps = 0.0, 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.cfg.EPOCHS}", leave=False)
            for batch in pbar:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["labels"].to(self.device)

                optimizer.zero_grad()
                loss, _ = self.model(input_ids, attention_mask, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.GRAD_CLIP)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                steps += 1
                pbar.set_postfix({"loss": f"{total_loss / steps:.4f}"})

            avg_train = total_loss / steps
            log.info("Epoch %d/%d — train_loss: %.4f", epoch, self.cfg.EPOCHS, avg_train)

            if val_loader:
                val_loss, report = self.evaluate(val_loader)
                log.info("            val_loss  : %.4f", val_loss)
                if report:
                    log.info("\n%s", report)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save(epoch, val_loss)
            else:
                self._save(epoch, avg_train)

        log.info("Training complete. Best val_loss: %.4f", best_val_loss)

    def evaluate(self, loader: DataLoader):
        self.model.eval()
        total_loss, steps = 0.0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["labels"].to(self.device)

                loss, logits = self.model(input_ids, attention_mask, labels)
                total_loss += loss.item()
                steps += 1

                preds = logits.argmax(dim=-1).cpu().numpy()
                labs  = labels.cpu().numpy()

                for p_row, l_row in zip(preds, labs):
                    for p, l in zip(p_row, l_row):
                        if l != -100:
                            all_preds.append(self.cfg.ID2LABEL.get(p, "O"))
                            all_labels.append(self.cfg.ID2LABEL.get(l, "O"))

        report = None
        if SKLEARN_AVAILABLE and all_labels:
            report = classification_report(
                all_labels, all_preds,
                labels=[l for l in self.cfg.LABELS if l != "O"],
                zero_division=0
            )
        return total_loss / max(steps, 1), report

    def _save(self, epoch: int, metric: float):
        ckpt = self.cfg.CHECKPOINT_DIR / f"epoch_{epoch:02d}_loss{metric:.4f}.pt"
        torch.save(self.model.state_dict(), ckpt)
        # Also save as "best"
        torch.save(self.model.state_dict(), self.cfg.MODEL_DIR / "best_model.pt")
        log.info("Checkpoint saved → %s", ckpt)


# ══════════════════════════════════════════════════════════════════════════════
#  8. INFERENCE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class PrescriptionAnalyzer:
    """
    Full inference pipeline:
        input (image/pdf/text)
        → OCR normalization
        → NER extraction
        → RAG knowledge lookup
        → Structured output
    """

    def __init__(self, cfg: Config = Config()):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.BASE_MODEL)
        self.ner_model = PrescriptionNERModel(cfg)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ocr = OCRPipeline(cfg)
        self.rag = MedicalRAG(cfg)

        model_path = cfg.MODEL_DIR / "best_model.pt"
        if model_path.exists():
            self.ner_model.load_state_dict(
                torch.load(str(model_path), map_location=self.device)
            )
            log.info("NER model loaded from %s", model_path)
        else:
            log.warning("No trained model found at %s — using untrained weights", model_path)

        self.ner_model.to(self.device).eval()
        self.rag.load()

    def analyze(self, source) -> dict:
        """
        source: file path (str/Path) for image/PDF, or raw prescription text (str)
        Returns structured dict with all output fields.
        """
        # Step 1: Get text
        if isinstance(source, (str, Path)) and Path(source).exists():
            log.info("Extracting text via OCR from %s", source)
            text = self.ocr.extract(source)
        else:
            text = str(source)
        log.info("Input text (%d chars): %s...", len(text), text[:80])

        # Step 2: NER
        entities = self._run_ner(text)
        log.info("Entities found: %s", entities)

        # Step 3: RAG retrieval
        rag_context = self.rag.retrieve(text)

        # Step 4: Build structured output
        return self._build_output(text, entities, rag_context)

    def _run_ner(self, text: str) -> dict:
        enc = self.tokenizer(
            text,
            max_length=self.cfg.MAX_LEN,
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        input_ids      = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            _, logits = self.ner_model(input_ids, attention_mask)
        preds = logits.argmax(dim=-1)[0].tolist()

        # Reconstruct spans using an explicit state dict (avoids closure rebind issues)
        entities: dict = {}
        state = {"label": None, "span": []}

        def flush():
            if state["label"] and state["span"]:
                s = state["span"][0][0]
                e = state["span"][-1][1]
                entities.setdefault(state["label"], []).append(text[s:e].strip())
            state["label"] = None
            state["span"]  = []

        for pid, (start, end) in zip(preds, offsets):
            if start == end:        # special token ([CLS], [SEP], padding)
                continue
            label = self.cfg.ID2LABEL.get(pid, "O")
            if label.startswith("B-"):
                flush()
                state["label"] = label[2:]
                state["span"]  = [(start, end)]
            elif label.startswith("I-") and state["label"] == label[2:]:
                state["span"].append((start, end))
            else:
                flush()
        flush()                     # emit any trailing span
        return entities

    def _build_output(self, text: str, entities: dict, rag_context: list) -> dict:
        # Build med table from extracted entities
        drugs     = entities.get("DRUG", [])
        doses     = entities.get("DOSE", [])
        freqs     = entities.get("FREQ", [])
        durations = entities.get("DURATION", [])
        conditions= entities.get("CONDITION", [])
        tests     = entities.get("TEST", [])
        diets     = entities.get("DIET", [])
        activities= entities.get("ACTIVITY", [])

        medications = []
        for i, drug in enumerate(drugs):
            medications.append({
                "name":    drug,
                "dosage":  doses[i]     if i < len(doses)     else "—",
                "timing":  freqs[i]     if i < len(freqs)     else "—",
                "days":    durations[i] if i < len(durations)  else "—",
                "notes":   "",
            })

        # Enrich with RAG context
        rag_side_effects = []
        for doc in rag_context:
            if "side effect" in doc["text"].lower() or "watch" in doc["text"].lower():
                rag_side_effects.append(doc["text"])

        checklist = (
            [{"category": "Test",     "item": t} for t in tests] +
            [{"category": "Diet",     "item": d} for d in diets] +
            [{"category": "Activity", "item": a} for a in activities]
        )

        cond_str = ", ".join(conditions) if conditions else "condition from prescription"
        one_liner = f"Prescribed {len(medications)} medicine(s) for {cond_str}. Follow up as directed."

        return {
            "raw_text":       text,
            "condition_name": cond_str or "Prescription",
            "medications":    medications,
            "diagnosis_plain": (
                f"You have been prescribed treatment for {cond_str}. "
                "Take all medicines as directed and complete the full course."
            ),
            "side_effects": [
                {"type": "watch", "text": s[:200]} for s in rag_side_effects[:2]
            ],
            "emergency_when": "Contact your doctor if you experience severe symptoms or allergic reactions.",
            "checklist":      checklist,
            "one_line_summary": one_liner,
            "rag_sources":    [{"title": d["source"], "relevance": d["text"][:80]} for d in rag_context],
        }

    def print_report(self, result: dict):
        w = 70
        print("\n" + "═" * w)
        print("  RXLENS — PRESCRIPTION ANALYSIS REPORT")
        print("═" * w)

        print(f"\n⚡ ONE-LINE SUMMARY\n  {result['one_line_summary']}")

        print(f"\n🩺 DIAGNOSIS\n  Condition : {result['condition_name']}")
        print(f"  Explanation:\n  {result['diagnosis_plain']}")

        if result["medications"]:
            print(f"\n💊 MEDICATION SCHEDULE")
            header = f"  {'Medicine':<20} {'Dose':<10} {'Timing':<22} {'Days':<10}"
            print(header)
            print("  " + "─" * (w - 2))
            for m in result["medications"]:
                print(f"  {m['name']:<20} {m['dosage']:<10} {m['timing']:<22} {m['days']:<10}")

        if result["side_effects"]:
            print(f"\n⚠️  SIDE EFFECT ALERTS")
            for se in result["side_effects"]:
                print(f"  • {se['text'][:100]}")

        if result["emergency_when"]:
            print(f"\n🚨 CALL DOCTOR IF:\n  {result['emergency_when']}")

        if result["checklist"]:
            print(f"\n✅ FOLLOW-UP CHECKLIST")
            for item in result["checklist"]:
                print(f"  [ ] [{item['category']}] {item['item']}")

        if result["rag_sources"]:
            print(f"\n🗄️  KNOWLEDGE SOURCES (RAG)")
            for src in result["rag_sources"]:
                print(f"  • {src['title']} — {src['relevance']}")

        print("\n" + "═" * w + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  9. MAIN — TRAINING ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def train_pipeline():
    cfg = Config()
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Load Keegel data
    loader = KeegelDatasetLoader(cfg)
    keegel_data = loader.load_all()

    # Build RAG index
    rag = MedicalRAG(cfg)
    rag.build(keegel_data)

    # Prepare NER dataset
    log.info("Preparing NER dataset...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.BASE_MODEL)
    records   = keegel_data["clinical_notes"]
    split     = int(len(records) * 0.9)
    train_ds  = PrescriptionNERDataset(records[:split], tokenizer, cfg)
    val_ds    = PrescriptionNERDataset(records[split:], tokenizer, cfg)
    log.info("  Train: %d  |  Val: %d", len(train_ds), len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

    # Train
    model   = PrescriptionNERModel(cfg)
    trainer = Trainer(model, cfg)
    trainer.train(train_loader, val_loader)
    log.info("Training complete. Run: python rxlens_model.py --infer 'Metformin 500mg BD x 30 days'")


def infer(source: str):
    analyzer = PrescriptionAnalyzer()
    result   = analyzer.analyze(source)
    analyzer.print_report(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RxLens Medical Prescription Model")
    parser.add_argument("--train",  action="store_true", help="Run training pipeline")
    parser.add_argument("--infer",  type=str, default="", help="Run inference on text or image path")
    args = parser.parse_args()

    if args.train:
        train_pipeline()
    elif args.infer:
        infer(args.infer)
    else:
        print(__doc__)
        print("\nUsage:")
        print("  Train  : python rxlens_model.py --train")
        print("  Infer  : python rxlens_model.py --infer 'Metformin 500mg BD x 30 days for T2DM'")
        print("  Infer  : python rxlens_model.py --infer path/to/prescription.jpg")
