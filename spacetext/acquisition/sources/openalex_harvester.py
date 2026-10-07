"""
OpenAlex API Harvester
======================
Harvests aerospace papers from Indian institutions via the OpenAlex API.
Uses ROR (Research Organization Registry) IDs and institution name filters
to ensure results are specific to Indian aerospace organisations.

Targeted Indian institutions:
  - ISRO (Indian Space Research Organisation) — ROR: 02v3cps24
  - IIT Bombay Aerospace Department — ROR: 01ctqnp55
  - IIT Madras — ROR: 03b8p8n71
  - IIT Kharagpur — ROR: 04m7fg108
  - DRDO (Defence Research & Development Organisation)
  - HAL (Hindustan Aeronautics Limited)
  - National Aerospace Laboratories
  - Physical Research Laboratory
"""

import time
import logging
import hashlib
from pathlib import Path
from typing import Iterator, Optional

import requests

logger = logging.getLogger(__name__)

OPENALEX_BASE_URL = "https://api.openalex.org/works"
REQUEST_DELAY_S = 1.0  # OpenAlex limit is 100,000 req/day

HEADERS = {
    "User-Agent": "SpaceText-Research-Bot/1.0 (mailto:spacetext@research.edu)"
}

# ROR-ID based filters for precise Indian institution targeting
# Format: filter=authorships.institutions.ror:https://ror.org/<id>
INDIAN_INSTITUTION_QUERIES = [
    # ISRO family (ROR IDs)
    {"type": "ror",    "value": "02v3cps24",  "label": "ISRO"},
    # IITs with aerospace departments
    {"type": "ror",    "value": "01ctqnp55",  "label": "IIT Bombay"},
    {"type": "ror",    "value": "03b8p8n71",  "label": "IIT Madras"},
    {"type": "ror",    "value": "04m7fg108",  "label": "IIT Kharagpur"},
    # Name-based searches for institutions without reliable RORs
    {"type": "search", "value": "National Aerospace Laboratories India",        "label": "NAL"},
    {"type": "search", "value": "Hindustan Aeronautics Limited",                "label": "HAL"},
    {"type": "search", "value": "Defence Research Development Organisation",    "label": "DRDO"},
    {"type": "search", "value": "Physical Research Laboratory Ahmedabad",       "label": "PRL"},
    {"type": "search", "value": "Indian Institute of Space Science Technology", "label": "IIST"},
    {"type": "search", "value": "Space Applications Centre ISRO",               "label": "SAC"},
    {"type": "search", "value": "Vikram Sarabhai Space Centre",                 "label": "VSSC"},
    {"type": "search", "value": "Liquid Propulsion Systems Centre",             "label": "LPSC"},
    {"type": "search", "value": "Indian Space Research",                        "label": "ISRO-General"},
]


class OpenAlexHarvester:
    """
    Harvests research papers from OpenAlex.
    """
    def __init__(
        self,
        output_dir: str = "./raw_data/openalex",
        download_pdfs: bool = True,
        max_records: int = 15000,
    ):
        self.output_dir = Path(output_dir)
        self.download_pdfs = download_pdfs
        self.max_records = max_records
        self.session: Optional[requests.Session] = None

    def connect(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        logger.info("OpenAlexHarvester: session ready.")

    def close(self) -> None:
        if self.session:
            self.session.close()
            self.session = None

    def harvest(self) -> Iterator[dict]:
        if self.session is None:
            raise RuntimeError("Call connect() first.")

        total_fetched = 0
        global_seen = set()

        logger.info(f"OpenAlex: starting Indian institution harvest (max_records={self.max_records})")

        for institution in INDIAN_INSTITUTION_QUERIES:
            if total_fetched >= self.max_records:
                break

            label = institution["label"]
            logger.info(f"OpenAlex: querying institution '{label}'...")

            if institution["type"] == "ror":
                # Use ROR ID filter for precise institution targeting
                filter_str = f"authorships.institutions.ror:https://ror.org/{institution['value']},has_abstract:true"
                params = {
                    "filter": filter_str,
                    "per-page": 100,
                    "cursor": "*",
                }
            else:
                # Use keyword search for institutions without ROR IDs
                params = {
                    "search": institution["value"],
                    "filter": "has_abstract:true",
                    "per-page": 100,
                    "cursor": "*",
                }

            page = 0
            institution_count = 0

            while total_fetched < self.max_records:
                time.sleep(REQUEST_DELAY_S)
                page += 1

                try:
                    resp = self.session.get(OPENALEX_BASE_URL, params=params, timeout=30)
                    resp.raise_for_status()
                except requests.RequestException as e:
                    logger.warning(f"OpenAlex [{label}]: request failed (page {page}): {e}")
                    time.sleep(10)
                    break

                data = resp.json()
                results = data.get("results", [])

                if not results:
                    break

                for work in results:
                    if total_fetched >= self.max_records:
                        break

                    doc_id = work.get("id", "").split("/")[-1]
                    if doc_id in global_seen:
                        continue
                    global_seen.add(doc_id)

                    title = work.get("title")
                    abstract = self._reconstruct_abstract(work.get("abstract_inverted_index", {}))

                    if not title or not abstract:
                        continue

                    # Best open access PDF url
                    pdf_url = None
                    oa = work.get("open_access") or {}
                    best_oa = work.get("best_oa_location") or {}
                    if oa.get("oa_url"):
                        pdf_url = oa["oa_url"]
                    elif best_oa.get("pdf_url"):
                        pdf_url = best_oa["pdf_url"]

                    year = str(work.get("publication_year", ""))
                    authors = [a.get("author", {}).get("display_name", "") for a in work.get("authorships", [])]
                    authors = [a for a in authors if a]

                    local_pdf_path = None
                    if self.download_pdfs and pdf_url:
                        local_pdf_path = self._download_pdf(pdf_url, doc_id)

                    yield {
                        "id": hashlib.sha256(doc_id.encode()).hexdigest()[:24],
                        "openalex_id": doc_id,
                        "title": title,
                        "abstract": abstract,
                        "authors": authors,
                        "year": year,
                        "source": "openalex",
                        "institution": label,
                        "source_url": work.get("id"),
                        "pdf_url": pdf_url,
                        "local_pdf_path": local_pdf_path,
                    }

                    total_fetched += 1
                    institution_count += 1

                # Pagination cursor
                next_cursor = data.get("meta", {}).get("next_cursor")
                if next_cursor and page < 10:  # max 10 pages per institution
                    params["cursor"] = next_cursor
                else:
                    break

            logger.info(f"OpenAlex [{label}]: {institution_count} records.")

        logger.info(f"OpenAlex: total harvested {total_fetched} Indian institution records.")

    def _reconstruct_abstract(self, inverted_index: dict) -> str:
        """OpenAlex returns abstracts as inverted indexes to bypass copyright. Reconstruct it."""
        if not inverted_index:
            return ""
        
        # Max index size to pre-allocate
        max_idx = max((idx for positions in inverted_index.values() for idx in positions), default=-1)
        if max_idx < 0:
            return ""
            
        words = [""] * (max_idx + 1)
        for word, positions in inverted_index.items():
            for pos in positions:
                words[pos] = word
                
        return " ".join(words)

    def _download_pdf(self, pdf_url: str, doc_id: str) -> Optional[str]:
        filename = hashlib.sha256(pdf_url.encode()).hexdigest()[:16] + ".pdf"
        local_path = self.output_dir / filename
        
        if local_path.exists():
            return str(local_path)
            
        try:
            time.sleep(REQUEST_DELAY_S)
            resp = self.session.get(pdf_url, timeout=30, stream=True)
            if resp.status_code == 200:
                with open(local_path, "wb") as f:
                    for chunk in resp.iter_content(8192):
                        f.write(chunk)
                logger.debug(f"OpenAlex: downloaded {local_path}")
                return str(local_path)
        except Exception as e:
            logger.debug(f"OpenAlex: PDF download failed for {doc_id}: {e}")
        return None
