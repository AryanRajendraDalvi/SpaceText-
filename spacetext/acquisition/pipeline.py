"""
SpaceText Acquisition Pipeline — Master Orchestrator
=====================================================
Runs all Indian aerospace data sources and feeds the
processed documents into GarudDB's 6-layer ingestion pipeline.

Sources (Indian-focused)
------------------------
  isro           — ISRO Publications portal + arXiv ISRO missions + OpenAlex ISRO institutions
  drdo           — DRDO Defence Science Journal (OAI-PMH) + DRDO website publications
  hal            — HAL (Hindustan Aeronautics Ltd) website + arXiv Tejas/ADA + OpenAlex HAL
  indian_journals— NISCAIR journals, Shodhganga theses, Indian private companies (AgniKul, Skyroot)
  arxiv          — arXiv broad aerospace harvest
  openalex       — OpenAlex Indian institution-filtered harvest

Output:  ./aerospace_taxonomy.adk   — GarudDB PersistentClient index
         ./raw_data/                 — Downloaded PDFs
         ./corpus/                   — Processed JSONL corpus
         ./acquisition_stats.json    — Summary report

Usage (run from project root):
    cd "d:\\downloads\\B.E. MAJOR PROJECT SPACETEXT"

    # All 5 sources (recommended — maximum corpus diversity)
    python -m spacetext.acquisition.pipeline

    # Specific sources only
    python -m spacetext.acquisition.pipeline --sources isro drdo arxiv
"""

import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Iterator, List, Optional
import concurrent.futures
import requests
import hashlib

# ── Setup logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log"),
    ],
)
logger = logging.getLogger("spacetext.pipeline")

# ── GarudDB (local editable install from d:\downloads\ADK) ────────────────────
try:
    import garuddb
    from garuddb import PersistentClient
    from garuddb.embeddings import GarudEmbedV5Function, DefaultEmbeddingFunction
    GARUDDB_AVAILABLE = True
    logger.info(f"GarudDB {garuddb.__version__} loaded.")
except ImportError:
    GARUDDB_AVAILABLE = False
    logger.error(
        "GarudDB not found. Run: pip install -e 'd:\\downloads\\ADK\\'"
    )

# ── Acquisition sources ────────────────────────────────────────────────────────
from spacetext.acquisition.sources.isro_scraper              import ISROScraper
from spacetext.acquisition.sources.drdo_connector             import DRDOConnector
from spacetext.acquisition.sources.arxiv_harvester            import ArXivHarvester
from spacetext.acquisition.sources.openalex_harvester         import OpenAlexHarvester
from spacetext.acquisition.sources.hal_scraper                import HALScraper
from spacetext.acquisition.sources.indian_aerospace_scraper   import IndianAerospaceJournalsScraper

# ── Processors ────────────────────────────────────────────────────────────────
from spacetext.acquisition.processors.pdf_extractor  import extract_pdf_sections
from spacetext.acquisition.processors.taxonomy_tagger import tag_document, get_primary_tag


# ── Configuration ──────────────────────────────────────────────────────────────
ALL_SOURCES = ["isro", "drdo", "hal", "indian_journals", "arxiv", "openalex"]

DEFAULT_CONFIG = {
    "raw_data_dir":          "./raw_data",
    "corpus_dir":            "./corpus",
    "garuddb_path":          "./aerospace_taxonomy.adk",
    "stats_path":            "./acquisition_stats.json",
    "arxiv_from":            "2015-01-01",
    "max_arxiv_docs":        50_000,
    "nasa_max_per_subject":  2000,    # NASA NTRS docs per subject category
    "use_hyperbolic":        True,    # GarudEmbedV5 Poincaré vectors
}


class AcquisitionPipeline:
    """
    Master pipeline: Source → Download → Extract → Tag → GarudDB.

    Each document flows through 5 stages:
      1. SOURCE    : API/scraper yields raw metadata + optional PDF path
      2. EXTRACT   : PDF → section-structured text (via pdf_extractor)
      3. TAG       : section text → taxonomy labels + depth (via taxonomy_tagger)
      4. CORPUS    : write JSONL record to corpus file (for SLM fine-tuning)
      5. GARUDDB   : add document to hyperbolic GarudDB collection
    """

    def __init__(self, config: Optional[dict] = None):
        self.cfg = {**DEFAULT_CONFIG, **(config or {})}
        Path(self.cfg["raw_data_dir"]).mkdir(parents=True, exist_ok=True)
        Path(self.cfg["corpus_dir"]).mkdir(parents=True, exist_ok=True)

        self.stats = {
            "total_ingested": 0,
            "by_source": {},
            "by_taxonomy_depth": {},
            "errors": 0,
            "start_time": time.time(),
        }

        # GarudDB collection
        self.collection = None
        if GARUDDB_AVAILABLE:
            self._init_garuddb()

    # ── GarudDB Setup ─────────────────────────────────────────────────────────

    def _init_garuddb(self):
        """Initialise or connect to the GarudDB aerospace collection."""
        client = PersistentClient(path=self.cfg["garuddb_path"])

        # Try GarudEmbedV5 first (64D Poincaré-native, already trained on hierarchical data)
        # Falls back to DefaultEmbeddingFunction (384D SentenceTransformer) if weights missing
        try:
            embed_fn = GarudEmbedV5Function()
            embed_name = "GarudEmbedV5"
        except Exception as e:
            logger.warning(f"GarudEmbedV5 not available ({e}). Using DefaultEmbeddingFunction.")
            embed_fn = DefaultEmbeddingFunction()
            embed_name = "DefaultEmbeddingFunction"

        # Create or get existing collection
        existing = client.list_collections()
        if "aerospace_taxonomy" in existing:
            self.collection = client.get_collection(
                "aerospace_taxonomy"
            )
            logger.info(f"GarudDB: resumed existing collection 'aerospace_taxonomy' ({embed_name})")
        else:
            self.collection = client.create_collection(
                name="aerospace_taxonomy",
                embedding_function=embed_fn,
                is_hyperbolic=self.cfg["use_hyperbolic"],
            )
            logger.info(f"GarudDB: created 'aerospace_taxonomy' collection ({embed_name}, hyperbolic={self.cfg['use_hyperbolic']})")

    # ── Main Run ──────────────────────────────────────────────────────────────

    def run(self, sources: List[str] = None):
        sources = sources or ALL_SOURCES
        logger.info(f"Pipeline starting. Sources: {sources}")
        corpus_path = Path(self.cfg["corpus_dir"]) / "aerospace_corpus.jsonl"
        raw_dir = self.cfg["raw_data_dir"]

        def _worker(raw_meta, source_name):
            try:
                # 1. Download PDF if needed
                if raw_meta.get("pdf_url") and not raw_meta.get("local_pdf_path"):
                    raw_meta["local_pdf_path"] = self._download_pdf(raw_meta["pdf_url"], raw_meta["id"], f"{raw_dir}/{source_name}")
                
                # 2. Process document
                record = self._process_document(raw_meta, source_name)
                return record
            except Exception as e:
                logger.debug(f"Worker error: {e}")
                return None

        with open(corpus_path, "a", encoding="utf-8") as corpus_file:
            for source_name in sources:
                logger.info(f"== Processing source: {source_name.upper()} ==")
                stream = self._get_source_stream(source_name)
                if stream is None:
                    continue

                source_count = 0
                batch = []
                
                # Fetching generator into a list so we can submit it
                # If stream is huge, this might take memory, but it's fine for 15k
                # Actually, OpenAlex chunks itself. But wait! OpenAlex is a generator.
                # If we submit all to thread pool, it will pull all from generator sequentially and submit.
                # This is perfect.
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    # We will submit tasks as they are generated
                    futures = set()
                    for raw_meta in stream:
                        futures.add(executor.submit(_worker, raw_meta, source_name))
                        
                        # Process completed futures to keep memory low and GarudDB fed
                        done_futures = [f for f in futures if f.done()]
                        for f in done_futures:
                            futures.remove(f)
                            record = f.result()
                            if not record:
                                self.stats["errors"] += 1
                                continue
                            
                            corpus_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                            corpus_file.flush()
                            
                            batch.append(record)
                            source_count += 1
                            self.stats["total_ingested"] += 1
                            
                            if len(batch) >= 32:
                                if self.collection:
                                    self._add_to_garuddb_batch(batch)
                                batch.clear()
                                
                            if source_count % 100 == 0:
                                logger.info(f"  [{source_name}] {source_count} docs ingested | Total: {self.stats['total_ingested']}")
                    
                    # Drain remaining
                    for future in concurrent.futures.as_completed(futures):
                        record = future.result()
                        if not record:
                            self.stats["errors"] += 1
                            continue
                            
                        corpus_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        corpus_file.flush()
                        
                        batch.append(record)
                        source_count += 1
                        self.stats["total_ingested"] += 1
                        
                        if len(batch) >= 32:
                            if self.collection:
                                self._add_to_garuddb_batch(batch)
                            batch.clear()

                        if source_count % 100 == 0:
                            logger.info(f"  [{source_name}] {source_count} docs ingested | Total: {self.stats['total_ingested']}")

                # Flush remaining batch
                if batch and self.collection:
                    self._add_to_garuddb_batch(batch)
                    
                self.stats["by_source"][source_name] = source_count
                logger.info(f"  [{source_name}] DONE — {source_count} documents.")

        self._save_stats()
        logger.info(f"Pipeline complete. Total: {self.stats['total_ingested']} docs. Errors: {self.stats['errors']}.")

    # ── Source Routing ────────────────────────────────────────────────────────

    def _get_source_stream(self, source_name: str) -> Optional[Iterator[dict]]:
        """Initialise and return the appropriate source stream."""
        raw_dir = self.cfg["raw_data_dir"]

        if source_name == "nasa":
            s = NTRSConnector(
                output_dir=f"{raw_dir}/nasa",
                download_pdfs=False,
                max_docs_per_subject=self.cfg["nasa_max_per_subject"],
            )
            s.connect()
            return s.stream_metadata()

        elif source_name == "isro":
            s = ISROScraper(output_dir=f"{raw_dir}/isro", download_pdfs=True)
            s.connect()
            def _isro_stream():
                yield from s.stream_documents()          # Official ISRO PDFs
                yield from s.stream_arxiv_isro()         # 500+ arXiv ISRO mission papers
                yield from s.stream_openalex_isro()      # OpenAlex ISRO institution papers
            return _isro_stream()

        elif source_name == "drdo":
            s = DRDOConnector(output_dir=f"{raw_dir}/drdo", download_pdfs=False)
            s.connect()
            # Combine both DRDO sub-sources into one stream
            def _drdo_stream():
                yield from s.stream_dsj(from_date="2010-01-01")
                yield from s.stream_website_publications()
            return _drdo_stream()

        elif source_name == "arxiv":
            s = ArXivHarvester(output_dir=f"{raw_dir}/arxiv", download_pdfs=False)
            s.connect()
            return s.harvest(from_date=self.cfg["arxiv_from"])

        elif source_name == "hal":
            s = HALScraper(output_dir=f"{raw_dir}/hal", download_pdfs=True)
            s.connect()
            return s.stream_documents()

        elif source_name == "indian_journals":
            s = IndianAerospaceJournalsScraper(output_dir=f"{raw_dir}/indian", download_pdfs=False)
            s.connect()
            return s.stream_documents()

        elif source_name == "openalex":
            s = OpenAlexHarvester(output_dir=f"{raw_dir}/openalex", download_pdfs=False, max_records=15000)
            s.connect()
            return s.harvest()

        else:
            logger.warning(f"Unknown source: '{source_name}'. Skipping.")
            return None

    # ── Document Processing ───────────────────────────────────────────────────


    def _download_pdf(self, pdf_url: str, doc_id: str, raw_dir: str) -> Optional[str]:
        if not pdf_url: return None
        filename = hashlib.sha256(pdf_url.encode()).hexdigest()[:16] + ".pdf"
        import os
        os.makedirs(raw_dir, exist_ok=True)
        local_path = Path(raw_dir) / filename
        if local_path.exists(): return str(local_path)
        try:
            resp = requests.get(pdf_url, timeout=30, stream=True)
            if resp.status_code == 200:
                with open(local_path, "wb") as f:
                    for chunk in resp.iter_content(8192): f.write(chunk)
                return str(local_path)
        except: pass
        return None

    def _add_to_garuddb_batch(self, records: List[dict]):
        docs, metas, ids = [], [], []
        for r in records:
            # FIX FLAW 2: Use full abstract (up to 3000 chars) for meaningful embeddings
            # Also append section text if available from PDFs
            section_text = " ".join(
                s.get("text", "") for s in r.get("sections", [])[:5]
                if s.get("section_type") in ("abstract", "introduction", "methodology", "results", "unknown")
            ).strip()
            abstract = r.get("abstract", "").strip()
            body = section_text[:2000] if section_text else ""
            doc_text = f"{r['title']}. {abstract[:3000]}" + (f" {body}" if body else "")
            if not doc_text.strip(): continue
            docs.append(doc_text[:6000])  # hard cap at 6k chars for embedding model
            metas.append({
                "taxonomy_label": r["taxonomy_label"][:64],
                "depth": r["taxonomy_depth"],
                "source": r["source"],
                "year": r.get("year", "")[:4],
                "path_len": len(r["taxonomy_path"]),
                "has_full_text": int(bool(section_text)),
            })
            ids.append(r["id"])
        
        if docs:
            self.collection.upsert(documents=docs, metadatas=metas, ids=ids)

    def _process_document(self, raw_meta: dict, source_name: str) -> Optional[dict]:
        """
        Full processing pipeline for one document:
          raw_meta → sections → taxonomy_tags → structured record
        """
        title    = (raw_meta.get("title") or "").strip()
        abstract = (raw_meta.get("abstract") or "").strip()

        if not title and not abstract:
            return None

        # Stage 2: Extract sections from PDF (if available) or use pre-extracted
        sections = raw_meta.get("sections", [])
        section_texts = [s.get("text", "") for s in sections]
        
        pdf_path = raw_meta.get("local_pdf_path")
        if pdf_path:
            try:
                extracted = extract_pdf_sections(pdf_path)
                if extracted:
                    sections = extracted
                    section_texts = [s["text"] for s in sections if s.get("text")]
            except Exception as e:
                logger.debug(f"PDF extraction failed for {pdf_path}: {e}")

        if not abstract and not sections:
            logger.debug(f"Skipping {source_name} doc {raw_meta.get('id')} — no abstract or PDF content")
            return None

        # Stage 3: Taxonomy tagging
        taxonomy_tags = tag_document(title, abstract, section_texts[:5])
        primary_label, primary_depth, taxonomy_path = (
            taxonomy_tags[0]["label"],
            taxonomy_tags[0]["depth"],
            taxonomy_tags[0]["path"],
        ) if taxonomy_tags else ("General Aerospace", 1, ["General Aerospace"])

        # Update depth stats
        depth_key = f"depth_{primary_depth}"
        self.stats["by_taxonomy_depth"][depth_key] = \
            self.stats["by_taxonomy_depth"].get(depth_key, 0) + 1

        # Build the SpaceText corpus record
        # This format is used directly for SLM fine-tuning (instruction → output)
        record = {
            "id":              raw_meta.get("id", ""),
            "title":           title,
            "abstract":        abstract,
            "sections":        sections[:25],   # FIX FLAW 3: raised from 10 to preserve more PDF content
            "taxonomy_label":  primary_label,
            "taxonomy_depth":  primary_depth,
            "taxonomy_path":   taxonomy_path,
            "taxonomy_tags":   taxonomy_tags[:3],
            "source":          source_name,
            "source_url":      raw_meta.get("source_url", ""),
            "year":            raw_meta.get("year", ""),
            "authors":         raw_meta.get("authors", []),
            # SLM fine-tuning instruction pairs
            "instruction_pairs": self._build_instruction_pairs(
                title, abstract, primary_label, taxonomy_path, sections
            ),
        }
        return record

    def _build_instruction_pairs(
        self,
        title: str,
        abstract: str,
        label: str,
        path: List[str],
        sections: list,
    ) -> List[dict]:
        """
        Generate instruction→output pairs for Qwen2.5 fine-tuning.
        Each pair is one training example in Alpaca format.
        """
        pairs = []

        # Task 1: Categorization
        pairs.append({
            "instruction": f"Categorize the following aerospace research document into the correct taxonomy category.\n\nTitle: {title}\n\nAbstract: {abstract[:800]}",
            "output": f"Category: {label}\nTaxonomy Path: {' > '.join(path)}\nDepth: {len(path)}",
            "task": "categorization",
        })

        # Task 2: Summarization — FIX FLAW 4 & 5: structured output, not a 300-char truncation
        if abstract and len(abstract) > 100:
            # Build a rich, structured summary as the training target
            section_highlights = ""
            if sections:
                highlights = [
                    s.get("heading", "").strip()
                    for s in sections[:5]
                    if s.get("heading") and s.get("section_type") not in ("references", "unknown")
                ]
                if highlights:
                    section_highlights = f"\nKey Sections: {', '.join(highlights)}"

            structured_output = (
                f"**Key Objective**: {abstract[:250].rstrip()}.\n"
                f"**Technical Domain**: {label} (Taxonomy: {' > '.join(path)})\n"
                f"**Summary**: {abstract[:600]}"
                + (f"\n{section_highlights}" if section_highlights else "")
            )
            pairs.append({
                "instruction": (
                    f"Summarize the following aerospace research document. Provide a structured summary "
                    f"with Key Objective, Technical Domain, and a concise technical Summary paragraph.\n\n"
                    f"Title: {title}\n\nAbstract: {abstract[:1200]}"
                ),
                "output": structured_output,
                "task": "summarization",
            })

        # Task 3: Multi-section synthesis (if we have full PDF sections)
        if sections and len(sections) >= 3:
            body_text = " ".join(
                s.get("text", "")[:300] for s in sections[:5]
                if s.get("section_type") in ("abstract", "introduction", "methodology", "results")
            ).strip()
            if body_text and len(body_text) > 200:
                pairs.append({
                    "instruction": (
                        f"You are given excerpts from a technical aerospace paper titled '{title}'.\n"
                        f"Synthesize the following sections into a coherent technical narrative:\n\n{body_text[:1500]}"
                    ),
                    "output": (
                        f"This paper presents work on {label}. {abstract[:400]}\n"
                        f"The research follows a {sections[1].get('heading','structured')} approach, "
                        f"focusing on contributions in {' > '.join(path)}."
                    ),
                    "task": "multi_section_synthesis",
                })

        # Task 4: Taxonomy path prediction
        if len(path) > 1:
            pairs.append({
                "instruction": f"Given this aerospace document title, predict the full taxonomy path from domain to specific system.\n\nTitle: {title}",
                "output": " > ".join(path),
                "task": "path_prediction",
            })

        return pairs

    # ── GarudDB Ingestion ─────────────────────────────────────────────────────

    def _add_to_garuddb(self, record: dict):
        """
        Add a processed document to GarudDB with depth metadata.
        Uses abstract + title as the document text.
        Metadata encodes taxonomy depth so GarudDB can do depth-filtered searches.
        """
        doc_text = f"{record['title']}. {record['abstract'][:500]}"
        if not doc_text.strip():
            return

        metadata = {
            "taxonomy_label": record["taxonomy_label"][:64],   # GarudDB metadata value limit
            "depth":          record["taxonomy_depth"],
            "source":         record["source"],
            "year":           record.get("year", "")[:4],
            "path_len":       len(record["taxonomy_path"]),
        }

        self.collection.add(
            documents=[doc_text],
            metadatas=[metadata],
            ids=[record["id"]],
        )

    # ── Stats ─────────────────────────────────────────────────────────────────

    def _save_stats(self):
        self.stats["elapsed_seconds"] = round(time.time() - self.stats["start_time"], 1)
        stats_path = self.cfg["stats_path"]
        with open(stats_path, "w") as f:
            json.dump(self.stats, f, indent=2)
        logger.info(f"Stats saved to {stats_path}")
        logger.info(json.dumps(self.stats, indent=2))


# ── CLI Entry Point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SpaceText Acquisition Pipeline — builds GarudDB aerospace index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All 5 sources (full corpus — recommended)
  python -m spacetext.acquisition.pipeline

  # Indian sources only
  python -m spacetext.acquisition.pipeline --sources isro drdo

  # NASA + arXiv only (quick large-scale run)
  python -m spacetext.acquisition.pipeline --sources nasa arxiv

  # All sources, custom GarudDB path
  python -m spacetext.acquisition.pipeline --garuddb-path ./my_index.adk
        """
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=ALL_SOURCES,
        default=ALL_SOURCES,
        help=(
            f"Data sources to run. Choices: {ALL_SOURCES}. "
            "Default: all 5 sources."
        ),
    )
    parser.add_argument(
        "--garuddb-path",
        default="./aerospace_taxonomy.adk",
        help="Path for GarudDB persistent index (default: ./aerospace_taxonomy.adk)",
    )
    parser.add_argument(
        "--no-hyperbolic",
        action="store_true",
        help="Use flat Euclidean index instead of Poincaré hyperbolic (not recommended)",
    )
    parser.add_argument(
        "--arxiv-from",
        default="2015-01-01",
        help="Start date for arXiv OAI-PMH harvest (YYYY-MM-DD, default: 2015-01-01)",
    )
    parser.add_argument(
        "--nasa-max-per-subject",
        type=int,
        default=2000,
        help="Max NASA NTRS documents per subject category (default: 2000)",
    )

    args = parser.parse_args()

    config = {
        "garuddb_path":          args.garuddb_path,
        "use_hyperbolic":        not args.no_hyperbolic,
        "arxiv_from":            args.arxiv_from,
        "nasa_max_per_subject":  args.nasa_max_per_subject,
    }

    pipeline = AcquisitionPipeline(config=config)
    pipeline.run(sources=args.sources)


if __name__ == "__main__":
    main()
