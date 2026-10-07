"""
arXiv Harvester
===============
Harvests aerospace-relevant papers from arXiv using two methods:
  1. arxiv Python library  — for small targeted queries (recent papers)
  2. OAI-PMH protocol     — for bulk metadata harvesting by date range

Relevant categories for SpaceText:
  astro-ph.IM  — Instrumentation and Methods for Astrophysics
  astro-ph.EP  — Earth and Planetary Astrophysics
  astro-ph.SR  — Solar and Stellar Astrophysics
  eess.SP      — Signal Processing (satellite/radar)
  cs.RO        — Robotics (space robotics)
  physics.space-ph — Space Physics
"""

import time
import logging
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, List, Optional
from xml.etree import ElementTree as ET

import requests

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
ARXIV_OAI_BASE   = "https://export.arxiv.org/oai2"
ARXIV_PDF_BASE   = "https://arxiv.org/pdf"
REQUEST_DELAY_S  = 3.0   # arXiv ToS requires ≥3s between requests

AEROSPACE_CATEGORIES = [
    "astro-ph.IM",
    "astro-ph.EP",
    "astro-ph.SR",
    "eess.SP",
    "cs.RO",
    "physics.space-ph",
]

OAI_NS = {
    "oai":    "http://www.openarchives.org/OAI/2.0/",
    "dc":     "http://purl.org/dc/elements/1.1/",
    "oai_dc": "http://www.openarchives.org/OAI/2.0/oai_dc/",
}

HEADERS = {
    "User-Agent": "SpaceText-Research-Bot/1.0 (academic; contact: spacetext@research.edu)",
}

# ── Indian Aerospace Specific Queries ─────────────────────────────────────────
# These supplement the broad OAI-PMH category harvest to ensure Indian
# aerospace research is well represented in the corpus.
INDIAN_AEROSPACE_QUERIES = [
    # Organisations
    "ISRO spacecraft",
    "Indian Space Research Organisation",
    "Vikram Sarabhai Space Centre",
    "DRDO missile guidance India",
    "Indian Institute of Space Science Technology",
    "IIST Thiruvananthapuram aerospace",
    "National Aerospace Laboratories India",
    # Missions
    "Chandrayaan Mangalyaan Aditya India",
    "PSLV GSLV launch India",
    "Gaganyaan India human spaceflight",
    "NavIC IRNSS India navigation",
    # Technologies
    "cryogenic engine India ISRO",
    "reusable launch vehicle India",
    "solid rocket motor India",
    "satellite attitude control India",
    "space debris India",
    # Private
    "AgniKul Skyroot India rocket",
    "Indian private space startup",
]


class ArXivHarvester:
    """
    Harvests arXiv papers for aerospace categories via OAI-PMH bulk protocol.

    OAI-PMH is the *compliant* way to bulk-harvest arXiv — it's the method
    arXiv officially recommends over scraping.

    Usage
    -----
    harvester = ArXivHarvester(output_dir="./raw_data/arxiv")
    harvester.connect()
    for record in harvester.harvest(from_date="2020-01-01", until_date="2024-12-31"):
        print(record["title"], record["arxiv_id"])
    harvester.close()
    """

    def __init__(
        self,
        output_dir: str = "./raw_data/arxiv",
        categories: Optional[List[str]] = None,
        download_pdfs: bool = False,   # PDFs are large; abstracts sufficient for phase 1
    ):
        self.output_dir = Path(output_dir)
        self.categories = categories or AEROSPACE_CATEGORIES
        self.download_pdfs = download_pdfs
        self.session: Optional[requests.Session] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def connect(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        logger.info("ArXivHarvester: session ready.")

    def close(self) -> None:
        if self.session:
            self.session.close()
            self.session = None

    # ── OAI-PMH Harvesting ────────────────────────────────────────────────────

    def harvest(
        self,
        from_date: str = "2015-01-01",
        until_date: Optional[str] = None,
    ) -> Iterator[dict]:
        """
        Yields metadata dicts for all records across all configured categories
        between from_date and until_date (ISO date strings: YYYY-MM-DD).
        Also runs the Indian aerospace keyword queries for targeted harvesting.
        """
        if self.session is None:
            raise RuntimeError("Call connect() before harvest().")

        if until_date is None:
            until_date = datetime.utcnow().strftime("%Y-%m-%d")

        # 1. Broad OAI-PMH category harvest
        for category in self.categories:
            logger.info(f"arXiv OAI-PMH: harvesting {category} from {from_date} to {until_date}")
            yield from self._harvest_category(category, from_date, until_date)

        # 2. Indian aerospace keyword-targeted harvest
        logger.info("arXiv: running Indian aerospace targeted queries...")
        yield from self.harvest_indian()

    def harvest_indian(self) -> Iterator[dict]:
        """
        Targeted keyword harvest for Indian aerospace papers.
        Uses the arXiv search API instead of OAI-PMH for keyword precision.
        """
        if self.session is None:
            raise RuntimeError("Call connect() first.")

        seen = set()
        total = 0

        for query in INDIAN_AEROSPACE_QUERIES:
            from urllib.parse import quote
            atom_url = (
                f"http://export.arxiv.org/api/query?"
                f"search_query=all:{quote(query)}&start=0&max_results=50"
            )
            try:
                time.sleep(REQUEST_DELAY_S)
                resp = self.session.get(atom_url, timeout=30)
                resp.raise_for_status()
            except Exception as e:
                logger.debug(f"arXiv Indian query failed for '{query}': {e}")
                continue

            try:
                from xml.etree import ElementTree as ET2
                root = ET2.fromstring(resp.content)
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                entries = root.findall("atom:entry", ns)
            except Exception as e:
                logger.debug(f"arXiv Indian parse failed: {e}")
                continue

            for entry in entries:
                arxiv_id = (entry.findtext("atom:id", namespaces=ns) or "").split("/abs/")[-1].strip()
                if not arxiv_id or arxiv_id in seen:
                    continue
                seen.add(arxiv_id)

                title = (entry.findtext("atom:title", namespaces=ns) or "").strip().replace("\n", " ")
                abstract = (entry.findtext("atom:summary", namespaces=ns) or "").strip().replace("\n", " ")
                published = (entry.findtext("atom:published", namespaces=ns) or "")[:10]
                authors = [
                    a.findtext("atom:name", namespaces=ns) or ""
                    for a in entry.findall("atom:author", ns)
                ]

                if not title or not abstract:
                    continue

                local_pdf = None
                if self.download_pdfs:
                    local_pdf = self._download_pdf(arxiv_id)

                doc_id = hashlib.sha256(arxiv_id.encode()).hexdigest()[:24]
                yield {
                    "id": doc_id,
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "year": published[:4],
                    "source": "arxiv",
                    "source_url": f"https://arxiv.org/abs/{arxiv_id}",
                    "local_pdf_path": local_pdf,
                }
                total += 1

        logger.info(f"arXiv Indian harvest: {total} papers ({len(seen)} unique).")


    def _harvest_category(
        self, category: str, from_date: str, until_date: str
    ) -> Iterator[dict]:
        """Paginate through OAI-PMH ListRecords for one category."""
        params = {
            "verb":            "ListRecords",
            "metadataPrefix":  "oai_dc",
            "set":             f"cs:{category}" if category.startswith("cs.") else f"physics:{category}" if category.startswith("physics.") else f"physics:{category}",
            "from":            from_date,
            "until":           until_date,
        }
        # arXiv uses set identifiers like: cs:cs.RO, physics:astro-ph.IM, etc.
        params["set"] = self._arxiv_set(category)

        resumption_token: Optional[str] = None
        page = 0

        while True:
            time.sleep(REQUEST_DELAY_S)
            page += 1

            if resumption_token:
                req_params = {"verb": "ListRecords", "resumptionToken": resumption_token}
            else:
                req_params = params

            try:
                resp = self.session.get(ARXIV_OAI_BASE, params=req_params, timeout=60)
                resp.raise_for_status()
            except requests.RequestException as e:
                logger.warning(f"arXiv OAI: request failed (page {page}): {e}. Retrying in 30s.")
                time.sleep(30)
                continue

            root = ET.fromstring(resp.content)
            records = root.findall(".//oai:record", OAI_NS)
            logger.info(f"arXiv OAI: {category} page {page} — {len(records)} records")

            for record in records:
                parsed = self._parse_oai_record(record, category)
                if parsed:
                    yield parsed

            # Check for resumption token (pagination)
            token_el = root.find(".//oai:resumptionToken", OAI_NS)
            if token_el is not None and token_el.text:
                resumption_token = token_el.text.strip()
            else:
                break  # no more pages

    # ── Parsers ───────────────────────────────────────────────────────────────

    def _parse_oai_record(self, record: ET.Element, category: str) -> Optional[dict]:
        """Extract clean metadata from a Dublin Core OAI-PMH record."""
        try:
            header = record.find("oai:header", OAI_NS)
            if header is None:
                return None

            status = header.get("status", "")
            if status == "deleted":
                return None

            oai_id = header.findtext("oai:identifier", namespaces=OAI_NS, default="")
            arxiv_id = oai_id.split(":")[-1] if ":" in oai_id else oai_id

            metadata = record.find(".//oai_dc:dc", OAI_NS)
            if metadata is None:
                return None

            title = (metadata.findtext("dc:title", namespaces=OAI_NS) or "").strip()
            abstract = (metadata.findtext("dc:description", namespaces=OAI_NS) or "").strip()
            date = (metadata.findtext("dc:date", namespaces=OAI_NS) or "")[:10]
            authors = [
                c.text.strip()
                for c in metadata.findall("dc:creator", OAI_NS)
                if c.text
            ]
            subjects = [
                s.text.strip()
                for s in metadata.findall("dc:subject", OAI_NS)
                if s.text
            ]

            if not title or not abstract:
                return None

            local_pdf_path = None
            if self.download_pdfs:
                local_pdf_path = self._download_pdf(arxiv_id)

            return {
                "id": hashlib.sha256(arxiv_id.encode()).hexdigest()[:24],
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "subjects": subjects,
                "year": date[:4],
                "date": date,
                "category": category,
                "source": "arxiv",
                "source_url": f"https://arxiv.org/abs/{arxiv_id}",
                "local_pdf_path": local_pdf_path,
            }
        except Exception as e:
            logger.debug(f"arXiv: failed to parse record: {e}")
            return None

    def _download_pdf(self, arxiv_id: str) -> Optional[str]:
        """Download PDF for a given arXiv ID. Returns local path."""
        safe_name = arxiv_id.replace("/", "_").replace(".", "_") + ".pdf"
        local_path = self.output_dir / "pdfs" / safe_name
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if local_path.exists():
            return str(local_path)

        url = f"{ARXIV_PDF_BASE}/{arxiv_id}.pdf"
        try:
            time.sleep(REQUEST_DELAY_S)
            resp = self.session.get(url, timeout=60, stream=True)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            return str(local_path)
        except requests.RequestException as e:
            logger.warning(f"arXiv: PDF download failed for {arxiv_id}: {e}")
            return None

    @staticmethod
    def _arxiv_set(category: str) -> str:
        """
        Convert category like 'astro-ph.IM' to the arXiv OAI set identifier.
        arXiv set naming: physics:astro-ph, cs:cs, etc.
        """
        if category.startswith("astro-ph") or category.startswith("physics"):
            prefix = "physics"
        elif category.startswith("cs"):
            prefix = "cs"
        elif category.startswith("eess"):
            prefix = "eess"
        else:
            prefix = "physics"
        
        # arXiv OAI-PMH uses colons for subclasses, e.g., physics:astro-ph:IM
        formatted_category = category.replace(".", ":")
        return f"{prefix}:{formatted_category}"
