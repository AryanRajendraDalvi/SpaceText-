"""
NASA News Scraper
=================
Scrapes NASA press releases and mission updates from nasa.gov/news.
Yields raw article text directly as "sections" since there are no PDFs.
"""

import time
import logging
import hashlib
from typing import Iterator, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

NASA_NEWS_URL = "https://www.nasa.gov/news/all-news/"
REQUEST_DELAY_S = 1.0

HEADERS = {
    "User-Agent": "SpaceText-Research-Bot/1.0"
}


class NasaNewsScraper:
    def __init__(self, output_dir: str = "./raw_data/nasa_news", max_records: int = 5000):
        self.output_dir = output_dir  # Not heavily used since we don't dl PDFs
        self.max_records = max_records
        self.session: Optional[requests.Session] = None

    def connect(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        logger.info("NasaNewsScraper: session ready.")

    def close(self) -> None:
        if self.session:
            self.session.close()
            self.session = None

    def harvest(self) -> Iterator[dict]:
        if self.session is None:
            raise RuntimeError("Call connect() first.")

        visited = set()
        total_fetched = 0
        current_url = NASA_NEWS_URL
        page = 0

        logger.info(f"NASA News: starting harvest (max_records={self.max_records})")

        while current_url and total_fetched < self.max_records:
            page += 1
            time.sleep(REQUEST_DELAY_S)

            try:
                resp = self.session.get(current_url, timeout=30)
                resp.raise_for_status()
            except requests.RequestException as e:
                logger.warning(f"NASA News: request failed (page {page}): {e}")
                break

            soup = BeautifulSoup(resp.text, "html.parser")
            
            # Find news links
            links = []
            for a in soup.select("a[href]"):
                href = a["href"]
                if "/news-release/" in href or "/feature/" in href or "/article/" in href:
                    if href.startswith("/"):
                        href = urljoin("https://www.nasa.gov", href)
                    if href not in visited:
                        visited.add(href)
                        links.append(href)

            if not links:
                logger.info(f"NASA News: no new links found on page {page}.")
                break

            for link in links:
                if total_fetched >= self.max_records:
                    break

                doc = self._scrape_article(link)
                if doc:
                    yield doc
                    total_fetched += 1

            # Check pagination
            next_link = soup.find("a", class_="next")
            if next_link and next_link.get("href"):
                current_url = next_link["href"]
            else:
                break

        logger.info(f"NASA News: harvested {total_fetched} records.")

    def _scrape_article(self, url: str) -> Optional[dict]:
        time.sleep(REQUEST_DELAY_S)
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            logger.debug(f"NASA News: failed to fetch {url}: {e}")
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        
        title_tag = soup.find("h1")
        if not title_tag:
            return None
            
        title = title_tag.get_text(strip=True)
        
        # Extract body text
        content_div = soup.find("div", class_="entry-content") or soup.find("article")
        if not content_div:
            return None
            
        paragraphs = [p.get_text(strip=True) for p in content_div.find_all("p") if p.get_text(strip=True)]
        
        if len(paragraphs) < 3:
            return None
            
        abstract = " ".join(paragraphs[:3])
        
        # We pre-format the sections array because there is no PDF to extract from!
        # The pipeline will use these pre-extracted sections.
        sections = [{"title": "Article Body", "text": " ".join(paragraphs)}]
        
        doc_id = hashlib.sha256(url.encode()).hexdigest()[:24]

        return {
            "id": doc_id,
            "title": title,
            "abstract": abstract,
            "authors": ["NASA"],
            "year": "2024",
            "source": "nasa_news",
            "source_url": url,
            "sections": sections,  # Pre-extracted text!
            "pdf_url": None,
            "local_pdf_path": None,
        }
