"""
ESA Publications Scraper
========================
Scrapes the European Space Agency's publicly available publications portal.
ESA's content is openly licensed for research use — source must be credited.

Target: https://www.esa.int/Newsroom/Publications
Respect: 1 request/second, identifies as a research bot via User-Agent.
"""

import time
import re
import logging
import hashlib
from pathlib import Path
from typing import Iterator, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

ESA_BASE_URL        = "https://www.esa.int"
ESA_PUBLICATIONS_URL = "https://www.esa.int/About_Us/ESA_Publications"
REQUEST_DELAY_S     = 1.0

HEADERS = {
    "User-Agent": "SpaceText-Research-Bot/1.0 (academic project; contact: spacetext@research.edu)",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}


class ESAScraper:
    """
    Scrapes ESA Newsroom publications and extracts metadata + PDF links.

    Usage
    -----
    scraper = ESAScraper(output_dir="./raw_data/esa")
    scraper.connect()
    for doc in scraper.stream_documents(max_docs=500):
        print(doc["title"], doc["local_pdf_path"])
    scraper.close()
    """

    def __init__(
        self,
        output_dir: str = "./raw_data/esa",
        start_url: str = ESA_PUBLICATIONS_URL,
        download_pdfs: bool = True,
        max_pages: int = 50,
    ):
        self.output_dir = Path(output_dir)
        self.start_url  = start_url
        self.download_pdfs = download_pdfs
        self.max_pages = max_pages
        self.session: Optional[requests.Session] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def connect(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        logger.info("ESAScraper: session ready.")

    def close(self) -> None:
        if self.session:
            self.session.close()
            self.session = None

    # ── Core Scraping ─────────────────────────────────────────────────────────

    def stream_documents(self, max_docs: Optional[int] = None) -> Iterator[dict]:
        """Yields metadata dicts for all ESA publications found."""
        if self.session is None:
            raise RuntimeError("Call connect() before stream_documents().")

        visited_links = set()
        doc_count = 0
        current_url = self.start_url
        page = 0

        while current_url and page < self.max_pages:
            if max_docs and doc_count >= max_docs:
                break

            logger.info(f"ESA: scraping page {page + 1}: {current_url}")
            time.sleep(REQUEST_DELAY_S)

            try:
                resp = self.session.get(current_url, timeout=30)
                resp.raise_for_status()
            except requests.RequestException as e:
                logger.warning(f"ESA: request failed: {e}")
                break

            soup = BeautifulSoup(resp.text, "html.parser")

            # Find publication links on this listing page
            pub_links = self._find_publication_links(soup)
            for link in pub_links:
                if max_docs and doc_count >= max_docs:
                    break
                if link in visited_links:
                    continue
                visited_links.add(link)

                doc = self._scrape_publication_page(link)
                if doc:
                    if self.download_pdfs and doc.get("pdf_url"):
                        doc["local_pdf_path"] = self._download_pdf(doc["pdf_url"], doc["id"])
                    else:
                        doc["local_pdf_path"] = None
                    yield doc
                    doc_count += 1

            # Find next page
            current_url = self._find_next_page(soup, current_url)
            page += 1

        logger.info(f"ESA: scraped {doc_count} documents total.")

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _find_publication_links(self, soup: BeautifulSoup) -> list:
        """Extract publication detail page URLs from a listing page."""
        links = []
        # ESA uses <a> tags within article/result blocks
        for tag in soup.select("a[href]"):
            href = tag.get("href", "")
            full_url = urljoin(ESA_BASE_URL, href)
            # Filter to only ESA content paths
            if "/Our_Activities/" in href or "/Enabling_Support/" in href or "/Science/" in href:
                if full_url not in links:
                    links.append(full_url)
        return links

    def _scrape_publication_page(self, url: str) -> Optional[dict]:
        """Scrape a single ESA publication detail page."""
        time.sleep(REQUEST_DELAY_S)
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.debug(f"ESA: failed to fetch {url}: {e}")
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        # Title
        title_tag = soup.find("h1") or soup.find("h2")
        title = title_tag.get_text(strip=True) if title_tag else ""
        if not title:
            return None

        # Abstract / body text — first few paragraphs
        body_paras = soup.select("div.article-body p, div.wysiwyg p, article p")
        abstract = " ".join(p.get_text(strip=True) for p in body_paras[:5])

        # Year — look for date patterns
        date_pattern = re.compile(r"\b(19|20)\d{2}\b")
        page_text = soup.get_text()
        years = date_pattern.findall(page_text)
        year = years[0] if years else ""

        # PDF link
        pdf_url = None
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.lower().endswith(".pdf"):
                pdf_url = urljoin(ESA_BASE_URL, href)
                break

        doc_id = hashlib.sha256(url.encode()).hexdigest()[:24]

        return {
            "id": doc_id,
            "title": title,
            "abstract": abstract[:2000],
            "year": year,
            "source": "esa",
            "source_url": url,
            "pdf_url": pdf_url,
        }

    def _download_pdf(self, pdf_url: str, doc_id: str) -> Optional[str]:
        """Download a PDF and return local path."""
        filename = hashlib.sha256(pdf_url.encode()).hexdigest()[:16] + ".pdf"
        local_path = self.output_dir / filename

        if local_path.exists():
            return str(local_path)

        try:
            time.sleep(REQUEST_DELAY_S)
            resp = self.session.get(pdf_url, timeout=60, stream=True)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            logger.debug(f"ESA: downloaded {local_path}")
            return str(local_path)
        except requests.RequestException as e:
            logger.warning(f"ESA: PDF download failed for {pdf_url}: {e}")
            return None

    def _find_next_page(self, soup: BeautifulSoup, current_url: str) -> Optional[str]:
        """Find the pagination 'Next' link."""
        for a in soup.find_all("a", string=re.compile(r"next|»|›", re.IGNORECASE)):
            href = a.get("href", "")
            if href:
                return urljoin(current_url, href)
        # Also check rel="next"
        next_link = soup.find("link", rel="next") or soup.find("a", rel="next")
        if next_link:
            return urljoin(current_url, next_link.get("href", ""))
        return None
