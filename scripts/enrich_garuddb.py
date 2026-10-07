"""
SpaceText GarudDB Enrichment Script
=====================================
One-time script to re-index the existing corpus into GarudDB with:
  1. Full abstracts (up to 3000 chars) instead of the old 500-char truncation
  2. Full PDF text for OpenAlex papers already downloaded to raw_data/openalex/
  3. A has_full_text metadata flag for filtering

Run from project root:
    python scripts/enrich_garuddb.py

This does NOT re-download anything. It re-processes what is already on disk.
"""

import json
import logging
import sys
from pathlib import Path

# Ensure the project root (parent of scripts/) is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("enrich.log"),
    ],
)
logger = logging.getLogger("spacetext.enrich")

CORPUS_PATH = "./corpus/aerospace_corpus.jsonl"
GARUDDB_PATH = "./aerospace_taxonomy.adk"
BATCH_SIZE = 64

try:
    from garuddb import PersistentClient
    GARUDDB_AVAILABLE = True
except ImportError:
    GARUDDB_AVAILABLE = False
    logger.error("GarudDB not found.")
    sys.exit(1)

from spacetext.acquisition.processors.pdf_extractor import extract_pdf_sections


def build_doc_text(record: dict, raw_data_dir: str = "./raw_data") -> tuple[str, bool]:
    """Build the richest possible document text for GarudDB indexing."""
    title = record.get("title", "").strip()
    abstract = record.get("abstract", "").strip()
    source = record.get("source", "")
    doc_id = record.get("id", "")
    section_text = ""
    has_full_text = False

    # Try to find and extract the PDF if it was downloaded
    if source == "openalex":
        # OpenAlex PDFs use sha256 hash naming
        import hashlib
        pdf_candidates = [
            Path(raw_data_dir) / "openalex" / f"{doc_id[:16]}.pdf",
        ]
        # Also check the source_url hash
        src_url = record.get("source_url", "")
        if src_url:
            url_hash = hashlib.sha256(src_url.encode()).hexdigest()[:16]
            pdf_candidates.append(Path(raw_data_dir) / "openalex" / f"{url_hash}.pdf")

        for pdf_path in pdf_candidates:
            if pdf_path.exists() and pdf_path.stat().st_size > 10000:  # skip stub PDFs < 10KB
                try:
                    sections = extract_pdf_sections(str(pdf_path))
                    if sections:
                        relevant = [
                            s.get("text", "") for s in sections[:8]
                            if s.get("section_type") in (
                                "abstract", "introduction", "methodology",
                                "results", "conclusion", "unknown"
                            )
                        ]
                        section_text = " ".join(relevant).strip()[:3000]
                        has_full_text = True
                        break
                except Exception as e:
                    logger.debug(f"PDF extraction failed for {pdf_path}: {e}")

    # Also use pre-extracted sections from the corpus record itself
    if not section_text and record.get("sections"):
        existing = [
            s.get("text", "") for s in record.get("sections", [])[:8]
            if s.get("section_type") in ("abstract", "introduction", "methodology", "results")
        ]
        section_text = " ".join(existing).strip()[:3000]
        if section_text:
            has_full_text = True

    # Compose the final indexing text
    body = f"{title}. {abstract[:3000]}"
    if section_text:
        body = f"{body} {section_text}"

    return body[:6000], has_full_text


def enrich(corpus_path: str, garuddb_path: str):
    logger.info(f"Loading GarudDB from {garuddb_path}...")
    client = PersistentClient(path=garuddb_path)
    collection = client.get_collection("aerospace_taxonomy")
    logger.info(f"Connected to collection 'aerospace_taxonomy' ({collection.count()} docs)")

    total = 0
    enriched = 0
    batch_docs, batch_metas, batch_ids = [], [], []

    with open(corpus_path, encoding="utf-8") as f:
        lines = f.readlines()

    logger.info(f"Processing {len(lines):,} corpus records...")

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        doc_text, has_full_text = build_doc_text(record)
        if not doc_text.strip():
            continue

        batch_docs.append(doc_text)
        batch_metas.append({
            "taxonomy_label": record.get("taxonomy_label", "General Aerospace")[:64],
            "depth":          record.get("taxonomy_depth", 1),
            "source":         record.get("source", "unknown"),
            "year":           str(record.get("year", ""))[:4],
            "path_len":       len(record.get("taxonomy_path", [])),
            "has_full_text":  int(has_full_text),
        })
        batch_ids.append(record["id"])
        total += 1
        if has_full_text:
            enriched += 1

        if len(batch_docs) >= BATCH_SIZE:
            try:
                collection.upsert(documents=batch_docs, metadatas=batch_metas, ids=batch_ids)
            except Exception as e:
                logger.warning(f"Batch upsert failed: {e}")
            batch_docs.clear()
            batch_metas.clear()
            batch_ids.clear()

            if total % 1000 == 0:
                logger.info(f"  {total:,} records processed | {enriched:,} with full text")

    # Final flush
    if batch_docs:
        try:
            collection.upsert(documents=batch_docs, metadatas=batch_metas, ids=batch_ids)
        except Exception as e:
            logger.warning(f"Final batch upsert failed: {e}")

    logger.info(f"\nEnrichment complete!")
    logger.info(f"  Total records processed : {total:,}")
    logger.info(f"  Records with full text  : {enriched:,} ({100*enriched/max(total,1):.1f}%)")
    logger.info(f"  GarudDB size            : {collection.count():,} docs")


if __name__ == "__main__":
    enrich(CORPUS_PATH, GARUDDB_PATH)
