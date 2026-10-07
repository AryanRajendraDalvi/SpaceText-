"""
HAL (Hindustan Aeronautics Limited) Scraper
============================================
Scrapes publicly available technical documents, annual reports, and R&D
publications from HAL's official website and OpenAlex institutional records.

Sources:
  - hal-india.co.in/MediaRoom and related pages
  - OpenAlex API filtered by institution "Hindustan Aeronautics"
  - arXiv papers mentioning HAL/TEJAS/ADA

License: Publicly available content; source credited.
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

HAL_BASE_URL   = "https://hal-india.co.in"
REQUEST_DELAY  = 1.5

HEADERS = {
    "User-Agent": "SpaceText-Research-Bot/1.0 (academic; contact: spacetext@research.edu)",
    "Accept": "text/html,application/xhtml+xml,application/xml,application/pdf",
}

# Pages on HAL website that may have technical content
HAL_PAGES = [
    "/MediaRoom/PressRelease",
    "/MediaRoom/Publications",
    "/AboutUs/RD",
    "/Products/Aircraft",
    "/Products/Helicopters",
    "/Products/Aero-Engines",
    "/Products/Avionics",
]

# arXiv queries for HAL & related Indian aerospace programmes
HAL_ARXIV_QUERIES = [
    "Tejas light combat aircraft",
    "HAL helicopter India aerospace",
    "Aeronautical Development Agency India",
    "ADA Tejas fly-by-wire",
    "Indian light combat aircraft",
    "Saras aircraft India",
    "DRDO ADA aircraft India",
]


class HALScraper:
    """
    Scrapes HAL publications and harvests related arXiv papers.

    Usage
    -----
    scraper = HALScraper(output_dir="./raw_data/hal")
    scraper.connect()
    for doc in scraper.stream_documents():
        print(doc["title"])
    """

    def __init__(
        self,
        output_dir: str = "./raw_data/hal",
        download_pdfs: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.download_pdfs = download_pdfs
        self.session: Optional[requests.Session] = None

    def connect(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        logger.info("HALScraper: session ready.")

    def close(self) -> None:
        if self.session:
            self.session.close()
            self.session = None

    def stream_documents(self) -> Iterator[dict]:
        """Yield documents from HAL website + arXiv HAL-related papers."""
        if self.session is None:
            raise RuntimeError("Call connect() before stream_documents().")

        yield from self._scrape_hal_website()
        yield from self._scrape_openalex_hal()
        yield from self._scrape_arxiv_hal()

    # ── HAL Website Scraper ────────────────────────────────────────────────────

    def _scrape_hal_website(self) -> Iterator[dict]:
        """Scrape HAL's public website for press releases, publications, PDFs."""
        seen = set()
        for page_path in HAL_PAGES:
            url = urljoin(HAL_BASE_URL, page_path)
            try:
                time.sleep(REQUEST_DELAY)
                resp = self.session.get(url, timeout=20)
                if resp.status_code != 200:
                    continue
            except Exception as e:
                logger.debug(f"HAL: failed to fetch {url}: {e}")
                continue

            soup = BeautifulSoup(resp.text, "html.parser")

            # Extract PDF links
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                if href.lower().endswith(".pdf"):
                    full_url = urljoin(HAL_BASE_URL, href)
                    if full_url in seen:
                        continue
                    seen.add(full_url)

                    title = a_tag.get_text(strip=True) or Path(href).stem.replace("_", " ").title()
                    doc_id = hashlib.sha256(full_url.encode()).hexdigest()[:24]

                    local_pdf = None
                    if self.download_pdfs:
                        local_pdf = self._download_pdf(full_url, doc_id)

                    abstract = (
                        f"HAL (Hindustan Aeronautics Limited) official document: {title}. "
                        f"Published by HAL, India's premier aerospace public sector undertaking, "
                        f"covering aircraft manufacturing, aero-engines, avionics, helicopters, "
                        f"and aerospace systems including the Tejas LCA programme."
                    )

                    yield {
                        "id": doc_id,
                        "title": title,
                        "abstract": abstract,
                        "year": self._extract_year(title + href),
                        "source": "hal",
                        "source_url": full_url,
                        "local_pdf_path": local_pdf,
                    }

            # Extract structured text articles/press releases
            articles = soup.find_all(["article", "div"], class_=re.compile(r"(press|news|content|article)", re.I))
            for art in articles[:20]:
                title_tag = art.find(["h1", "h2", "h3", "h4"])
                if not title_tag:
                    continue
                title = title_tag.get_text(strip=True)
                if not title or title in seen:
                    continue
                seen.add(title)

                # Get body text
                paras = art.find_all("p")
                body = " ".join(p.get_text(strip=True) for p in paras[:5])
                if not body:
                    continue

                doc_id = hashlib.sha256(title.encode()).hexdigest()[:24]
                yield {
                    "id": doc_id,
                    "title": title,
                    "abstract": body[:2000],
                    "year": self._extract_year(title),
                    "source": "hal",
                    "source_url": url,
                    "local_pdf_path": None,
                }

        logger.info("HAL: website scraping complete.")

    # ── OpenAlex for HAL ──────────────────────────────────────────────────────

    def _scrape_openalex_hal(self) -> Iterator[dict]:
        """Query OpenAlex for papers authored by HAL-affiliated researchers."""
        OPENALEX_URL = "https://api.openalex.org/works"
        # Search for works from Indian aerospace institutions
        queries = [
            "Hindustan Aeronautics",
            "HAL Bangalore aerospace",
            "Aeronautical Development Agency Bangalore",
        ]
        seen = set()

        for query in queries:
            params = {
                "search": query,
                "filter": "has_abstract:true",
                "per-page": 50,
                "cursor": "*",
            }
            pages_fetched = 0
            while pages_fetched < 3:
                try:
                    time.sleep(1.0)
                    resp = self.session.get(OPENALEX_URL, params=params, timeout=30)
                    resp.raise_for_status()
                except Exception as e:
                    logger.debug(f"HAL OpenAlex query failed: {e}")
                    break

                data = resp.json()
                for work in data.get("results", []):
                    work_id = work.get("id", "").split("/")[-1]
                    if work_id in seen:
                        continue
                    seen.add(work_id)

                    title = work.get("title", "")
                    abstract = self._reconstruct_abstract(work.get("abstract_inverted_index", {}))
                    if not title or not abstract:
                        continue

                    doc_id = hashlib.sha256(work_id.encode()).hexdigest()[:24]
                    yield {
                        "id": doc_id,
                        "title": title,
                        "abstract": abstract,
                        "year": str(work.get("publication_year", "")),
                        "source": "hal",
                        "source_url": work.get("id", ""),
                        "local_pdf_path": None,
                    }

                next_cursor = data.get("meta", {}).get("next_cursor")
                if next_cursor:
                    params["cursor"] = next_cursor
                    pages_fetched += 1
                else:
                    break

        logger.info(f"HAL: OpenAlex harvest complete ({len(seen)} records).")

    # ── arXiv Fallback for HAL/ADA ────────────────────────────────────────────

    def _scrape_arxiv_hal(self) -> Iterator[dict]:
        """Query arXiv for papers related to HAL/ADA/Tejas programmes."""
        seen = set()
        for query in HAL_ARXIV_QUERIES:
            atom_url = (
                f"http://export.arxiv.org/api/query?"
                f"search_query=all:{requests.utils.quote(query)}&start=0&max_results=30"
            )
            try:
                time.sleep(REQUEST_DELAY)
                resp = self.session.get(atom_url, timeout=20)
                resp.raise_for_status()
            except Exception as e:
                logger.debug(f"HAL arXiv query failed for '{query}': {e}")
                continue

            import xml.etree.ElementTree as ET
            try:
                root = ET.fromstring(resp.content)
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                entries = root.findall("atom:entry", ns)
            except Exception:
                continue

            for entry in entries:
                arxiv_id = (entry.findtext("atom:id", namespaces=ns) or "").split("/abs/")[-1].strip()
                if not arxiv_id or arxiv_id in seen:
                    continue
                seen.add(arxiv_id)

                title = (entry.findtext("atom:title", namespaces=ns) or "").strip().replace("\n", " ")
                abstract = (entry.findtext("atom:summary", namespaces=ns) or "").strip().replace("\n", " ")
                published = (entry.findtext("atom:published", namespaces=ns) or "")[:10]

                if not title:
                    continue

                doc_id = hashlib.sha256(arxiv_id.encode()).hexdigest()[:24]
                yield {
                    "id": doc_id,
                    "title": title,
                    "abstract": abstract,
                    "year": published[:4],
                    "source": "hal",
                    "source_url": f"https://arxiv.org/abs/{arxiv_id}",
                    "local_pdf_path": None,
                }

        logger.info(f"HAL: arXiv harvest complete ({len(seen)} papers).")

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _extract_year(self, text: str) -> str:
        m = re.search(r"\b(19|20)\d{2}\b", text)
        return m.group(0) if m else ""

    def _reconstruct_abstract(self, inverted_index: dict) -> str:
        if not inverted_index:
            return ""
        max_idx = max((i for positions in inverted_index.values() for i in positions), default=-1)
        if max_idx < 0:
            return ""
        words = [""] * (max_idx + 1)
        for word, positions in inverted_index.items():
            for pos in positions:
                words[pos] = word
        return " ".join(words)

    def _download_pdf(self, url: str, doc_id: str) -> Optional[str]:
        filename = doc_id + ".pdf"
        local_path = self.output_dir / filename
        if local_path.exists():
            return str(local_path)
        try:
            time.sleep(REQUEST_DELAY)
            resp = self.session.get(url, timeout=60, stream=True)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            return str(local_path)
        except Exception as e:
            logger.debug(f"HAL: PDF download failed for {url}: {e}")
            return None
