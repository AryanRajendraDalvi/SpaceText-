"""
ISRO Publications Downloader
============================
Downloads publicly available annual reports, technical brochures, and
mission documents from ISRO's Publications portal.
Also performs comprehensive arXiv and OpenAlex harvesting for ISRO-related content.

Source: https://www.isro.gov.in/Publications.html
License: Publicly available; source must be credited.
"""

import re
import time
import logging
import hashlib
from pathlib import Path
from typing import Iterator, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

ISRO_BASE_URL    = "https://www.isro.gov.in"
ISRO_PUB_URL     = "https://www.isro.gov.in/publications.html"
REQUEST_DELAY_S  = 1.5   # polite crawl delay

HEADERS = {
    "User-Agent": "SpaceText-Research-Bot/1.0 (academic; contact: spacetext@research.edu)",
    "Accept": "text/html,application/pdf",
    "Referer": "https://www.isro.gov.in/",
}

# Publication categories available on ISRO portal
ISRO_CATEGORIES = [
    "Annual Reports",
    "Brochures",
    "Technical Reports",
    "ISRO Journals",
    "Newsletters",
    "Satellite & Launch Vehicle Brochures",
]


class ISROScraper:
    """
    Downloads ISRO publications (PDFs) and extracts metadata.
    Also harvests ISRO-related papers from arXiv and OpenAlex.

    Usage
    -----
    scraper = ISROScraper(output_dir="./raw_data/isro")
    scraper.connect()
    for doc in scraper.stream_documents():
        print(doc["title"], doc["local_pdf_path"])
    scraper.close()
    """

    def __init__(
        self,
        output_dir: str = "./raw_data/isro",
        download_pdfs: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.download_pdfs = download_pdfs
        self.session: Optional[requests.Session] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def connect(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        logger.info("ISROScraper: session ready.")

    def close(self) -> None:
        if self.session:
            self.session.close()
            self.session = None

    # ── Core Scraping ─────────────────────────────────────────────────────────

    def stream_documents(self) -> Iterator[dict]:
        """Yield metadata + local paths for all ISRO publications."""
        if self.session is None:
            raise RuntimeError("Call connect() before stream_documents().")

        logger.info(f"ISRO: fetching publications index from {ISRO_PUB_URL}")
        time.sleep(REQUEST_DELAY_S)

        try:
            resp = self.session.get(ISRO_PUB_URL, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"ISRO: failed to fetch publications page: {e}")
            return

        soup = BeautifulSoup(resp.text, "html.parser")
        pdf_links = self._extract_pdf_links(soup)

        logger.info(f"ISRO: found {len(pdf_links)} PDF links")

        for title, pdf_url, year, category in pdf_links:
            doc_id = hashlib.sha256(pdf_url.encode()).hexdigest()[:24]

            local_pdf_path = None
            if self.download_pdfs:
                local_pdf_path = self._download_pdf(pdf_url, doc_id)

            abstract = (
                f"ISRO {category} document: {title}. "
                f"Published by the Indian Space Research Organisation. Year: {year or 'unknown'}."
            )

            yield {
                "id": doc_id,
                "title": title,
                "abstract": abstract,
                "year": year,
                "category": category,
                "source": "isro",
                "source_url": pdf_url,
                "local_pdf_path": local_pdf_path,
            }

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _extract_pdf_links(self, soup: BeautifulSoup) -> list:
        """
        Returns list of (title, full_pdf_url, year, category) tuples.
        ISRO publications page uses <a href="...pdf"> tags with descriptive text.
        """
        results = []
        year_re = re.compile(r"\b(19|20)\d{2}\b")

        current_category = "General"

        for tag in soup.find_all(["h2", "h3", "h4", "a"]):
            if tag.name in ("h2", "h3", "h4"):
                heading_text = tag.get_text(strip=True)
                for cat in ISRO_CATEGORIES:
                    if cat.lower() in heading_text.lower():
                        current_category = cat
                        break

            elif tag.name == "a":
                href = tag.get("href", "")
                if not href.lower().endswith(".pdf"):
                    continue

                full_url = urljoin(ISRO_BASE_URL, href)
                title = tag.get_text(strip=True)
                if not title:
                    title = Path(href).stem.replace("_", " ").replace("-", " ").title()

                year_matches = year_re.findall(title + href)
                year = year_matches[0] if year_matches else ""

                results.append((title, full_url, year, current_category))

        return results

    def _download_pdf(self, url: str, doc_id: str) -> Optional[str]:
        """Download a PDF to local storage. Returns path or None."""
        filename = doc_id + ".pdf"
        local_path = self.output_dir / filename

        if local_path.exists():
            return str(local_path)

        try:
            time.sleep(REQUEST_DELAY_S)
            resp = self.session.get(url, timeout=60, stream=True)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            logger.debug(f"ISRO: downloaded {local_path}")
            return str(local_path)
        except requests.RequestException as e:
            logger.warning(f"ISRO: PDF download failed for {url}: {e}")
            return None

    # ── arXiv Fallback for ISRO-related papers ────────────────────────────────

    def stream_arxiv_isro(self) -> Iterator[dict]:
        """
        Expanded ISRO arXiv fallback with comprehensive mission-specific queries.
        Targets 500+ papers across all major ISRO programs.
        """
        import xml.etree.ElementTree as ET

        ISRO_QUERIES = [
            # Missions
            "Chandrayaan lunar India",
            "Chandrayaan-2 orbit India",
            "Chandrayaan-3 lander India",
            "Mangalyaan Mars orbit India",
            "Aditya-L1 solar wind India",
            "ASTROSAT X-ray India",
            "RISAT radar satellite India",
            "CartoSat remote sensing India",
            # Launch vehicles
            "PSLV launch vehicle India",
            "GSLV cryogenic India",
            "GSLV Mark III India",
            "LVM3 reusable India",
            "Scramjet ISRO India",
            # Technologies
            "NavIC IRNSS positioning India",
            "IRNSS navigation India",
            "Gaganyaan human spaceflight India",
            "ISRO spacecraft attitude control",
            "ISRO propulsion cryogenic",
            "ISRO space debris",
            "VSSC rocket India",
            "SAC space applications India",
            "Indian Space Research Organisation",
            # Organisations
            "Vikram Sarabhai Space Centre",
            "IIST aerospace Thiruvananthapuram",
            "Physical Research Laboratory India space",
            "Space Applications Centre Ahmedabad",
            "URSC satellite India Bangalore",
            # Applications
            "INSAT communication satellite India",
            "GSAT communication India",
            "IRS remote sensing India",
            "RLV-TD reusable India",
        ]

        seen_ids = set()
        total = 0

        for query_term in ISRO_QUERIES:
            atom_url = (
                f"http://export.arxiv.org/api/query?"
                f"search_query=all:{requests.utils.quote(query_term)}&start=0&max_results=50"
            )
            try:
                time.sleep(REQUEST_DELAY_S)
                resp = self.session.get(atom_url, timeout=20)
                resp.raise_for_status()
            except Exception as e:
                logger.debug(f"ISRO arXiv query failed for '{query_term}': {e}")
                continue

            try:
                root = ET.fromstring(resp.content)
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                entries = root.findall("atom:entry", ns)
            except Exception as e:
                logger.debug(f"ISRO arXiv parse failed: {e}")
                continue

            for entry in entries:
                arxiv_id = (entry.findtext("atom:id", namespaces=ns) or "").split("/abs/")[-1].strip()
                if not arxiv_id or arxiv_id in seen_ids:
                    continue
                seen_ids.add(arxiv_id)

                title = (entry.findtext("atom:title", namespaces=ns) or "").strip().replace("\n", " ")
                abstract = (entry.findtext("atom:summary", namespaces=ns) or "").strip().replace("\n", " ")
                published = (entry.findtext("atom:published", namespaces=ns) or "")[:10]
                authors = [
                    a.findtext("atom:name", namespaces=ns) or ""
                    for a in entry.findall("atom:author", ns)
                ]

                if not title:
                    continue

                doc_id = hashlib.sha256(arxiv_id.encode()).hexdigest()[:24]
                yield {
                    "id":         doc_id,
                    "title":      title,
                    "abstract":   abstract,
                    "authors":    authors,
                    "year":       published[:4],
                    "source":     "isro",
                    "source_url": f"https://arxiv.org/abs/{arxiv_id}",
                    "local_pdf_path": None,
                }
                total += 1

        logger.info(f"ISRO arXiv: yielded {total} papers ({len(seen_ids)} unique).")

    def stream_openalex_isro(self) -> Iterator[dict]:
        """
        Query OpenAlex for papers from ISRO-affiliated institutions.
        """
        if self.session is None:
            raise RuntimeError("Call connect() first.")

        OPENALEX_URL = "https://api.openalex.org/works"
        ISRO_INSTITUTION_QUERIES = [
            "Indian Space Research Organisation",
            "Vikram Sarabhai Space Centre",
            "ISRO Satellite Centre ISAC",
            "Space Applications Centre ISRO",
            "Liquid Propulsion Systems Centre",
            "Physical Research Laboratory Ahmedabad",
            "National Remote Sensing Centre NRSC",
        ]

        seen = set()
        total = 0

        for institution_query in ISRO_INSTITUTION_QUERIES:
            params = {
                "search": institution_query,
                "filter": "has_abstract:true",
                "per-page": 100,
                "cursor": "*",
            }
            pages = 0
            while pages < 5:
                try:
                    time.sleep(REQUEST_DELAY_S)
                    resp = self.session.get(OPENALEX_URL, params=params, timeout=30)
                    resp.raise_for_status()
                except Exception as e:
                    logger.debug(f"ISRO OpenAlex query failed: {e}")
                    break

                data = resp.json()
                for work in data.get("results", []):
                    work_id = work.get("id", "").split("/")[-1]
                    if work_id in seen:
                        continue
                    seen.add(work_id)

                    title = work.get("title", "")
                    abstract_inv = work.get("abstract_inverted_index", {})
                    if not abstract_inv:
                        continue

                    max_idx = max((i for positions in abstract_inv.values() for i in positions), default=-1)
                    if max_idx < 0:
                        continue
                    words = [""] * (max_idx + 1)
                    for word, positions in abstract_inv.items():
                        for pos in positions:
                            words[pos] = word
                    abstract = " ".join(words)

                    if not title or not abstract:
                        continue

                    pdf_url = None
                    oa = work.get("open_access") or {}
                    best_oa = work.get("best_oa_location") or {}
                    if oa.get("oa_url"):
                        pdf_url = oa["oa_url"]
                    elif best_oa.get("pdf_url"):
                        pdf_url = best_oa["pdf_url"]

                    local_pdf = None
                    if self.download_pdfs and pdf_url:
                        doc_id_hash = hashlib.sha256(pdf_url.encode()).hexdigest()[:24]
                        local_pdf = self._download_pdf(pdf_url, doc_id_hash)

                    doc_id = hashlib.sha256(work_id.encode()).hexdigest()[:24]
                    yield {
                        "id": doc_id,
                        "title": title,
                        "abstract": abstract,
                        "year": str(work.get("publication_year", "")),
                        "source": "isro",
                        "source_url": work.get("id", ""),
                        "local_pdf_path": local_pdf,
                    }
                    total += 1

                next_cursor = data.get("meta", {}).get("next_cursor")
                if next_cursor:
                    params["cursor"] = next_cursor
                    pages += 1
                else:
                    break

        logger.info(f"ISRO OpenAlex: harvested {total} records.")
