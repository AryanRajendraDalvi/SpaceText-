"""
NASA Technical Reports Server (NTRS) Connector
================================================
Uses the official NASA STI OpenAPI at ntrs.nasa.gov/api to search and
download publicly available aerospace research documents.

Docs: https://ntrs.nasa.gov/api/swagger-ui
Rate limit: no hard cap, but we self-throttle to 2 req/s to be polite.
"""

import time
import logging
import hashlib
from pathlib import Path
from typing import Iterator, List, Optional

import requests

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
NTRS_SEARCH_URL  = "https://ntrs.nasa.gov/api/citations/search"
NTRS_DOWNLOAD_URL = "https://ntrs.nasa.gov/api/citations/{doc_id}/downloads/{filename}"
REQUEST_DELAY_S  = 0.5      # seconds between API calls
PAGE_SIZE        = 100      # max results per page (NTRS supports up to 100)

# AIAA-aligned subject categories we care about for SpaceText
AEROSPACE_SUBJECTS = [
    "Space Transportation",
    "Propulsion",
    "Astrodynamics",
    "Spacecraft Design",
    "Aerodynamics",
    "Structures",
    "Guidance and Control",
    "Communications",
    "Remote Sensing",
    "Life Sciences",
    "Astronomy",
]

HEADERS = {
    "User-Agent": "SpaceText-Research-Bot/1.0 (academic project; contact: spacetext@research.edu)",
    "Accept": "application/json",
}


class NTRSConnector:
    """
    Streams aerospace research documents from the NASA NTRS API.

    Usage
    -----
    connector = NTRSConnector(output_dir="./raw_data/nasa")
    connector.connect()
    for meta in connector.stream_metadata(max_docs=5000):
        print(meta["title"], meta["local_pdf_path"])
    connector.close()
    """

    def __init__(
        self,
        output_dir: str = "./raw_data/nasa",
        subjects: Optional[List[str]] = None,
        max_docs_per_subject: int = 2000,
        download_pdfs: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.subjects = subjects or AEROSPACE_SUBJECTS
        self.max_docs_per_subject = max_docs_per_subject
        self.download_pdfs = download_pdfs
        self.session: Optional[requests.Session] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def connect(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        # Verify the API is reachable
        resp = self.session.get(NTRS_SEARCH_URL, params={"page": {"size": 1}}, timeout=10)
        resp.raise_for_status()
        logger.info("NTRSConnector: connected to NASA NTRS API.")

    def close(self) -> None:
        if self.session:
            self.session.close()
            self.session = None

    # ── Core Search ────────────────────────────────────────────────────────────

    def _search_page(self, subject: str, from_offset: int) -> dict:
        """Execute a single paginated POST search request."""
        payload = {
            "query": subject,
            "page": {
                "size": PAGE_SIZE,
                "from": from_offset,
            },
            "facets": {
                "subject": [subject],
            },
        }
        time.sleep(REQUEST_DELAY_S)
        resp = self.session.post(NTRS_SEARCH_URL, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def stream_metadata(self, max_docs: Optional[int] = None) -> Iterator[dict]:
        """
        Yields metadata dicts for every document found across all subjects.
        Each dict contains: id, title, authors, abstract, subject, year,
        local_pdf_path (if downloaded).
        """
        if self.session is None:
            raise RuntimeError("Call connect() before stream_metadata().")

        total_yielded = 0

        for subject in self.subjects:
            logger.info(f"NTRS: querying subject='{subject}'")
            from_offset = 0
            subject_count = 0

            while True:
                if max_docs and total_yielded >= max_docs:
                    return

                try:
                    data = self._search_page(subject, from_offset)
                except requests.RequestException as e:
                    logger.warning(f"NTRS: request failed for subject '{subject}' at offset {from_offset}: {e}")
                    break

                hits = data.get("results", [])
                if not hits:
                    break  # exhausted this subject

                for hit in hits:
                    if max_docs and total_yielded >= max_docs:
                        return
                    if subject_count >= self.max_docs_per_subject:
                        break

                    meta = self._parse_hit(hit, subject)
                    if meta is None:
                        continue

                    # Download PDF if requested
                    if self.download_pdfs:
                        meta["local_pdf_path"] = self._download_pdf(hit)
                    else:
                        meta["local_pdf_path"] = None

                    yield meta
                    total_yielded += 1
                    subject_count += 1

                if subject_count >= self.max_docs_per_subject:
                    break

                from_offset += PAGE_SIZE
                if len(hits) < PAGE_SIZE:
                    break  # no more pages

            logger.info(f"NTRS: fetched {subject_count} docs for subject='{subject}'")

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _parse_hit(self, hit: dict, subject: str) -> Optional[dict]:
        """Extract clean metadata from an NTRS citation result."""
        try:
            doc_id = hit.get("id") or hit.get("_id")
            if not doc_id:
                return None

            title = hit.get("title", "").strip()
            abstract = hit.get("abstract", "").strip()
            
            year = ""
            pubs = hit.get("publications", [])
            if pubs and pubs[0].get("publicationDate"):
                year = pubs[0]["publicationDate"][:4]
            elif hit.get("created"):
                year = hit.get("created")[:4]

            authors = []
            for affil in hit.get("authorAffiliations", []):
                name = affil.get("meta", {}).get("author", {}).get("name", "")
                if name:
                    authors.append(name)
                    
            downloads = hit.get("downloads", [])

            return {
                "id": str(doc_id),
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "year": year,
                "subject": subject,
                "source": "nasa_ntrs",
                "source_url": f"https://ntrs.nasa.gov/citations/{doc_id}",
                "downloads": downloads,
            }
        except Exception as e:
            logger.debug(f"NTRS: failed to parse hit: {e}")
            return None

    def _download_pdf(self, hit: dict) -> Optional[str]:
        """Download the first available PDF for a document. Returns local path."""
        doc_id = hit.get("id") or hit.get("_id")
        downloads = hit.get("downloads", [])

        pdf_dl = next((d for d in downloads if "pdf" in d.get("mimetype", "").lower() or d.get("name", "").endswith(".pdf")), None)
        if not pdf_dl:
            return None

        links = pdf_dl.get("links", {})
        filename = links.get("pdf", links.get("original", "")).split("/")[-1]
        if not filename:
            filename = pdf_dl.get("name", f"{doc_id}.pdf")

        # Stable local filename based on doc_id hash
        safe_name = hashlib.sha256(str(doc_id).encode()).hexdigest()[:16] + ".pdf"
        local_path = self.output_dir / safe_name

        if local_path.exists():
            return str(local_path)  # already cached

        url = NTRS_DOWNLOAD_URL.format(doc_id=doc_id, filename=filename)
        try:
            time.sleep(REQUEST_DELAY_S)
            resp = self.session.get(url, timeout=60, stream=True)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.debug(f"NTRS: downloaded {local_path}")
            return str(local_path)
        except requests.RequestException as e:
            logger.warning(f"NTRS: PDF download failed for {doc_id}: {e}")
            return None
