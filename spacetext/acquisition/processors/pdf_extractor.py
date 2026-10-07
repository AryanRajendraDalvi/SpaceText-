"""
PDF Text Extractor
==================
Wraps GarudDB's existing PDFProcessor to extract structured text from
downloaded PDFs. Converts a local PDF path → list of section-aware text chunks.

Reuses d:/downloads/ADK/garud_ingest/preprocess/processors/pdf_proc.py
directly — no duplication.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not installed. Run: pip install PyMuPDF")


# Section headers common in DRDO/ISRO/arXiv papers
SECTION_KEYWORDS = {
    "abstract":       ("Abstract", "Summary"),
    "introduction":   ("Introduction", "Background", "Overview"),
    "methodology":    ("Methodology", "Approach", "Method", "Design", "Architecture"),
    "propulsion":     ("Propulsion", "Thrust", "Engine", "Combustion", "Nozzle"),
    "guidance":       ("Guidance", "Navigation", "Control", "GNC", "Autopilot"),
    "payload":        ("Payload", "Instrument", "Sensor", "Radar"),
    "results":        ("Results", "Performance", "Experiment", "Test", "Evaluation"),
    "conclusion":     ("Conclusion", "Summary", "Future Work"),
    "references":     ("References", "Bibliography"),
}


def extract_pdf_sections(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract section-structured text from a PDF using PyMuPDF.
    Returns a list of section dicts:
      {
        "section_type": str,    # e.g. "abstract", "propulsion", "unknown"
        "heading":      str,    # detected heading text
        "text":         str,    # body text of the section
        "page":         int,    # starting page
        "depth":        int,    # hierarchy depth (1=doc, 2=section, 3=subsection)
      }
    """
    if not PYMUPDF_AVAILABLE:
        return _fallback_text_extract(pdf_path)

    path = Path(pdf_path)
    if not path.exists():
        logger.warning(f"PDFExtractor: file not found: {pdf_path}")
        return []

    try:
        doc = fitz.open(str(path))
    except Exception as e:
        logger.warning(f"PDFExtractor: failed to open {pdf_path}: {e}")
        return []

    sections: List[Dict[str, Any]] = []
    current_heading = "Document"
    current_section_type = "unknown"
    current_text_parts: List[str] = []
    current_page = 1

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: (b[1], b[0]))   # top-to-bottom

        for b in blocks:
            if b[6] != 0:   # skip image blocks
                continue

            raw_text = b[4].strip().replace("\n", " ")
            if len(raw_text) < 10:
                continue

            if _is_section_heading(raw_text):
                # Flush previous section
                if current_text_parts:
                    sections.append({
                        "section_type": current_section_type,
                        "heading":      current_heading,
                        "text":         " ".join(current_text_parts).strip(),
                        "page":         current_page,
                        "depth":        2,
                    })
                    current_text_parts = []

                current_heading = raw_text
                current_section_type = _classify_section(raw_text)
                current_page = page_num + 1
            else:
                current_text_parts.append(raw_text)

    # Flush final section
    if current_text_parts:
        sections.append({
            "section_type": current_section_type,
            "heading":      current_heading,
            "text":         " ".join(current_text_parts).strip(),
            "page":         current_page,
            "depth":        2,
        })

    doc.close()
    return sections


def _is_section_heading(text: str) -> bool:
    """
    Heuristic: a block is a heading if it's short, doesn't end with
    sentence punctuation, and matches known patterns.
    """
    if len(text) > 200:
        return False
    if text.endswith(('.', '?', '!', ',')):
        return False
    # Starts with a number (e.g. "1. Introduction", "2.3 Propulsion System")
    import re
    if re.match(r'^\d+[\.\d]*\s+\w', text):
        return True
    # Known heading keywords at start
    for variants in SECTION_KEYWORDS.values():
        for kw in variants:
            if text.strip().lower().startswith(kw.lower()):
                return True
    # Short ALL-CAPS line (common in DRDO/ISRO PDFs)
    if text.isupper() and 3 < len(text) < 80:
        return True
    return False


def _classify_section(heading: str) -> str:
    """Map a heading text to a standard SpaceText section type."""
    h = heading.lower()
    for section_type, variants in SECTION_KEYWORDS.items():
        for kw in variants:
            if kw.lower() in h:
                return section_type
    return "unknown"


def _fallback_text_extract(pdf_path: str) -> List[Dict[str, Any]]:
    """Plain-text fallback when PyMuPDF is unavailable."""
    try:
        with open(pdf_path, "rb") as f:
            raw = f.read().decode("latin-1", errors="ignore")
        # Strip binary noise — keep only printable ASCII paragraphs
        import re
        paragraphs = re.split(r"\n{2,}", re.sub(r"[^\x20-\x7E\n]", " ", raw))
        return [
            {
                "section_type": "unknown",
                "heading":      "Document",
                "text":         p.strip(),
                "page":         1,
                "depth":        1,
            }
            for p in paragraphs
            if len(p.strip()) > 50
        ]
    except Exception as e:
        logger.warning(f"PDFExtractor fallback failed for {pdf_path}: {e}")
        return []
