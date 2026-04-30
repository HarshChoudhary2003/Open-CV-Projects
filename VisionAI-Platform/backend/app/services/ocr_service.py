"""
VisionAI Platform - OCR Text Extraction Service
Uses EasyOCR (GPU-capable) with Tesseract fallback.
"""

import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import easyocr
    _HAS_EASY = True
except ImportError:
    _HAS_EASY = False

try:
    import pytesseract
    _HAS_TESS = True
except ImportError:
    _HAS_TESS = False


class OCRResult:
    def __init__(self, text: str, confidence: float, bbox: Tuple):
        self.text = text
        self.confidence = confidence
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.timestamp = time.time()

    def to_dict(self) -> dict:
        x1, y1, x2, y2 = self.bbox
        return {
            "text": self.text,
            "confidence": round(self.confidence, 3),
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        }


class OCRService:
    _instance: Optional["OCRService"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialised = False
        return cls._instance

    def initialise(self, use_gpu: bool = False):
        if self._initialised:
            return

        if _HAS_EASY:
            self.reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
            self.backend = "easyocr"
            print("[OCRService] Backend: EasyOCR")
        elif _HAS_TESS:
            self.backend = "tesseract"
            print("[OCRService] Backend: Tesseract")
        else:
            self.backend = "none"
            print("[OCRService] No OCR backend available. Install easyocr or pytesseract.")

        self._initialised = True

    # ── Processing ────────────────────────────────────────────────

    def extract(self, frame: np.ndarray, min_confidence: float = 0.3
                ) -> Tuple[np.ndarray, List[OCRResult]]:
        annotated = frame.copy()
        results: List[OCRResult] = []

        if self.backend == "none":
            return annotated, results

        if self.backend == "easyocr":
            raw = self.reader.readtext(frame, detail=1, paragraph=False)
            for (pts, text, conf) in raw:
                if conf < min_confidence or len(text.strip()) < 2:
                    continue
                x1, y1 = int(pts[0][0]), int(pts[0][1])
                x2, y2 = int(pts[2][0]), int(pts[2][1])
                r = OCRResult(text.strip(), conf, (x1, y1, x2, y2))
                results.append(r)
                self._draw(annotated, r)

        elif self.backend == "tesseract":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            data = pytesseract.image_to_data(
                gray, output_type=pytesseract.Output.DICT
            )
            for i, text in enumerate(data["text"]):
                text = text.strip()
                conf = int(data["conf"][i])
                if not text or conf < 30:
                    continue
                x, y, w, h = (data["left"][i], data["top"][i],
                               data["width"][i], data["height"][i])
                r = OCRResult(text, conf / 100, (x, y, x + w, y + h))
                results.append(r)
                self._draw(annotated, r)

        return annotated, results

    def _draw(self, frame: np.ndarray, r: OCRResult):
        x1, y1, x2, y2 = r.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 1)
        cv2.putText(frame, r.text, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1, cv2.LINE_AA)


def get_ocr_service() -> OCRService:
    svc = OCRService()
    svc.initialise()
    return svc
