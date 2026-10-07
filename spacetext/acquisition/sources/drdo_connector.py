"""
DRDO Publications Connector
============================
Harvests open-access research from DRDO (Defence Research & Development Organisation)
via two channels:

  1. Defence Science Journal (DSJ) — OJS-based, OAI-PMH compliant
     Endpoint: http://publications.drdo.gov.in/ojs/index.php/dsj/oai
     Topics: aerospace, electronics, materials, propulsion, guidance systems

  2. DRDO Website Publications — Annual reports, Tech Focus newsletters,
     mission brochures from https://www.drdo.gov.in

All content is open-access and published by a Government of India body.
Source must be credited in research output.

OJS OAI-PMH reference:
  https://pkp.sfu.ca/ojs/docs/userguide/2.3.3/publishingReviewingSubmissions.html
"""

import re
import time
import logging
import hashlib
from pathlib import Path
from typing import Iterator, List, Optional
from xml.etree import ElementTree as ET
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
DSJ_OAI_ENDPOINT  = "https://publicationsdrdo.in/index.php/dsj/oai"
DRDO_PUB_URL      = "https://www.drdo.gov.in/publications"
DRDO_BASE_URL     = "https://www.drdo.gov.in"
REQUEST_DELAY_S   = 1.5   # polite throttle

OAI_NS = {
    "oai":    "http://www.openarchives.org/OAI/2.0/",
    "dc":     "http://purl.org/dc/elements/1.1/",
    "oai_dc": "http://www.openarchives.org/OAI/2.0/oai_dc/",
}

HEADERS = {
    "User-Agent": "SpaceText-Research-Bot/1.0 (academic; B.E. Final Year Project)",
    "Accept": "text/html,application/xml,application/pdf",
}

# DRDO subject areas relevant to SpaceText / aerospace
DSJ_SUBJECTS = [
    "Aerospace",
    "Propulsion",
    "Guidance",
    "Navigation",
    "Electronics",
    "Radar",
    "Missile",
    "Satellite",
    "Defence",
]


class DRDOConnector:
    """
    Streams research documents from DRDO's open-access publications.

    Covers two sources:
      - DSJ (Defence Science Journal) via OAI-PMH
      - DRDO website publications (Annual Reports, Tech Focus, brochures) via scraping

    Usage
    -----
    connector = DRDOConnector(output_dir="./raw_data/drdo")
    connector.connect()

    # Stream DSJ journal articles (metadata + abstracts)
    for doc in connector.stream_dsj(from_date="2015-01-01"):
        print(doc["title"])

    # Stream annual reports / brochures (PDFs)
    for doc in connector.stream_website_publications():
        print(doc["title"], doc["local_pdf_path"])

    connector.close()
    """

    def __init__(
        self,
        output_dir: str = "./raw_data/drdo",
        download_pdfs: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.download_pdfs = download_pdfs
        self.session: Optional[requests.Session] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def connect(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "dsj").mkdir(exist_ok=True)
        (self.output_dir / "website").mkdir(exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        logger.info("DRDOConnector: session ready.")

    def close(self) -> None:
        if self.session:
            self.session.close()
            self.session = None

    # ══════════════════════════════════════════════════════════════════════════
    # SOURCE 1: Defence Science Journal — OAI-PMH
    # ══════════════════════════════════════════════════════════════════════════

    def stream_dsj(
        self,
        from_date: str = "2010-01-01",
        until_date: Optional[str] = None,
        max_records: Optional[int] = None,
    ) -> Iterator[dict]:
        """
        Harvest Defence Science Journal articles via OAI-PMH ListRecords.
        Returns metadata dicts with: id, title, abstract, authors, year,
        subjects, source_url, local_pdf_path.
        """
        if self.session is None:
            raise RuntimeError("Call connect() first.")

        # First verify the OAI endpoint is alive
        try:
            resp = self._oai_request({"verb": "Identify"})
            repo_name = resp.findtext(".//oai:repositoryName", namespaces=OAI_NS) or "DRDO DSJ"
            logger.info(f"DRDOConnector: OAI endpoint alive — {repo_name}")
        except Exception as e:
            logger.warning(f"DRDOConnector: OAI endpoint unreachable ({e}). Falling back to web scraping.")
            yield from self._stream_dsj_fallback()
            return

        params = {
            "verb":           "ListRecords",
            "metadataPrefix": "oai_dc",
            "from":           from_date,
        }
        if until_date:
            params["until"] = until_date

        resumption_token: Optional[str] = None
        total = 0
        page = 0

        while True:
            if max_records and total >= max_records:
                break

            page += 1
            time.sleep(REQUEST_DELAY_S)

            req_params = (
                {"verb": "ListRecords", "resumptionToken": resumption_token}
                if resumption_token
                else params
            )

            try:
                root = self._oai_request(req_params)
            except Exception as e:
                logger.warning(f"DRDO DSJ: OAI request failed (page {page}): {e}")
                time.sleep(10)
                continue

            records = root.findall(".//oai:record", OAI_NS)
            logger.info(f"DRDO DSJ: page {page} — {len(records)} records")

            for record in records:
                if max_records and total >= max_records:
                    break
                parsed = self._parse_dsj_record(record)
                if parsed:
                    if self.download_pdfs and parsed.get("pdf_url"):
                        parsed["local_pdf_path"] = self._download_pdf(
                            parsed["pdf_url"],
                            parsed["id"],
                            subfolder="dsj",
                        )
                    yield parsed
                    total += 1

            # Pagination
            token_el = root.find(".//oai:resumptionToken", OAI_NS)
            if token_el is not None and token_el.text and token_el.text.strip():
                resumption_token = token_el.text.strip()
            else:
                break

        logger.info(f"DRDO DSJ: harvested {total} records.")

    def _oai_request(self, params: dict) -> ET.Element:
        resp = self.session.get(DSJ_OAI_ENDPOINT, params=params, timeout=30)
        resp.raise_for_status()
        return ET.fromstring(resp.content)

    def _parse_dsj_record(self, record: ET.Element) -> Optional[dict]:
        """Parse a single Dublin Core OAI-PMH record from DSJ."""
        try:
            header = record.find("oai:header", OAI_NS)
            if header is None or header.get("status") == "deleted":
                return None

            oai_id = header.findtext("oai:identifier", namespaces=OAI_NS, default="")
            doc_id = hashlib.sha256(oai_id.encode()).hexdigest()[:24]

            dc = record.find(".//oai_dc:dc", OAI_NS)
            if dc is None:
                return None

            title    = (dc.findtext("dc:title",       namespaces=OAI_NS) or "").strip()
            abstract = (dc.findtext("dc:description", namespaces=OAI_NS) or "").strip()
            date     = (dc.findtext("dc:date",        namespaces=OAI_NS) or "")[:10]
            authors  = [
                c.text.strip()
                for c in dc.findall("dc:creator", OAI_NS)
                if c.text
            ]
            subjects = [
                s.text.strip()
                for s in dc.findall("dc:subject", OAI_NS)
                if s.text
            ]

            # PDF link is usually in dc:identifier as a URL
            pdf_url = None
            for ident in dc.findall("dc:identifier", OAI_NS):
                val = (ident.text or "").strip()
                if val.startswith("http") and val.lower().endswith(".pdf"):
                    pdf_url = val
                    break
                # Sometimes the article landing page URL is here
                if val.startswith("http") and "dsj" in val and not pdf_url:
                    pdf_url = None  # keep looking for actual PDF

            if not title:
                return None

            return {
                "id":          doc_id,
                "oai_id":      oai_id,
                "title":       title,
                "abstract":    abstract[:3000],
                "authors":     authors,
                "subjects":    subjects,
                "year":        date[:4],
                "date":        date,
                "source":      "drdo_dsj",
                "source_url":  f"http://publications.drdo.gov.in/ojs/index.php/dsj",
                "pdf_url":     pdf_url,
                "local_pdf_path": None,
            }
        except Exception as e:
            logger.debug(f"DRDO DSJ: parse error: {e}")
            return None

    def _stream_dsj_fallback(self) -> Iterator[dict]:
        """
        Fallback: scrape DSJ article listing pages directly when OAI is down.
        Tries common DSJ HTML pages for article metadata.
        """
        logger.info("DRDO DSJ: using HTML fallback scraper.")
        base = "https://publicationsdrdo.in/index.php/dsj/issue/archive"
        try:
            resp = self.session.get(base, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            logger.error(f"DRDO DSJ fallback also failed: {e}")
            return

        soup = BeautifulSoup(resp.text, "html.parser")
        issue_links = [
            urljoin(base, a["href"])
            for a in soup.select("a[href]")
            if "/issue/view/" in a.get("href", "")
        ]

        for issue_url in issue_links[:20]:  # limit fallback depth
            time.sleep(REQUEST_DELAY_S)
            try:
                iresp = self.session.get(issue_url, timeout=30)
                iresp.raise_for_status()
            except Exception:
                continue

            isoup = BeautifulSoup(iresp.text, "html.parser")
            for article in isoup.select("div.article-summary, div.tocTitle"):
                title_tag = article.find("a")
                if not title_tag:
                    continue
                title = title_tag.get_text(strip=True)
                article_url = urljoin(base, title_tag.get("href", ""))

                doc_id = hashlib.sha256(article_url.encode()).hexdigest()[:24]
                yield {
                    "id":           doc_id,
                    "title":        title,
                    "abstract":     f"DRDO Defence Science Journal article: {title}",
                    "authors":      [],
                    "subjects":     [],
                    "year":         "",
                    "source":       "drdo_dsj_fallback",
                    "source_url":   article_url,
                    "pdf_url":      None,
                    "local_pdf_path": None,
                }

    # ══════════════════════════════════════════════════════════════════════════
    # SOURCE 2: DRDO Website — Annual Reports, Tech Focus, Brochures
    # ══════════════════════════════════════════════════════════════════════════

    def stream_website_publications(self) -> Iterator[dict]:
        """
        Scrapes PDF links from DRDO's main website publications section.
        Covers Annual Reports, Technology Focus, lab brochures, etc.
        """
        if self.session is None:
            raise RuntimeError("Call connect() first.")

        # Known stable PDF-hosting sub-paths on drdo.gov.in
        # Most of these are currently blocking non-browser agents or geofenced.
        # We rely on DSJ (above) for the bulk of DRDO's public innovations.
        pub_urls = [
            # "https://www.drdo.gov.in/annual-report",
            # "https://www.drdo.gov.in/drdo-technology-focus",
            # "https://www.drdo.gov.in/publications",
            # "https://www.drdo.gov.in/drdo-newsletter",
        ]

        for url in pub_urls:
            logger.info(f"DRDO website: scraping {url}")
            time.sleep(REQUEST_DELAY_S)

            try:
                resp = self.session.get(url, timeout=30)
                resp.raise_for_status()
            except requests.RequestException as e:
                logger.warning(f"DRDO website: failed to fetch {url}: {e}")
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            pdf_links = self._extract_pdf_links_from_page(soup, url)

            logger.info(f"DRDO website: found {len(pdf_links)} PDFs on {url}")

            for title, pdf_url, year, category in pdf_links:
                doc_id = hashlib.sha256(pdf_url.encode()).hexdigest()[:24]

                local_pdf_path = None
                if self.download_pdfs:
                    local_pdf_path = self._download_pdf(pdf_url, doc_id, subfolder="website")

                # Synthesize an abstract from metadata (DRDO PDFs rarely have machine-readable abstracts)
                abstract = (
                    f"DRDO publication: {title}. "
                    f"Category: {category}. "
                    f"Published by Defence Research and Development Organisation, India. "
                    f"Year: {year or 'unknown'}. "
                    "Covers defence and aerospace technology development."
                )

                yield {
                    "id":           doc_id,
                    "title":        title,
                    "abstract":     abstract,
                    "authors":      ["DRDO"],
                    "subjects":     [category, "Defence", "Aerospace"],
                    "year":         year,
                    "source":       "drdo_website",
                    "source_url":   pdf_url,
                    "local_pdf_path": local_pdf_path,
                }

    def _extract_pdf_links_from_page(
        self, soup: BeautifulSoup, base_url: str
    ) -> List[tuple]:
        """Returns list of (title, full_pdf_url, year, category) from a page."""
        results = []
        year_re = re.compile(r"\b(19|20)\d{2}[-–]\d{2,4}\b|\b(19|20)\d{2}\b")
        category = self._infer_category_from_url(base_url)

        for a in soup.find_all("a", href=True):
            href = a["href"]
            if not href.lower().endswith(".pdf"):
                continue

            full_url = urljoin(base_url, href)
            title = a.get_text(strip=True)
            if not title:
                title = Path(href).stem.replace("_", " ").replace("-", " ").title()

            # Extract year
            combined = title + " " + href
            year_matches = year_re.findall(combined)
            year = (year_matches[0] if year_matches else ("", ""))
            year = year[0] if isinstance(year, tuple) else year

            results.append((title, full_url, year, category))

        return results

    def _infer_category_from_url(self, url: str) -> str:
        if "annual" in url:
            return "Annual Report"
        if "technology-focus" in url or "tech" in url:
            return "Technology Focus"
        if "newsletter" in url:
            return "Newsletter"
        return "General Publication"

    # ── PDF Downloader ─────────────────────────────────────────────────────────

    def _download_pdf(self, url: str, doc_id: str, subfolder: str = "") -> Optional[str]:
        """Download a PDF. Returns local path or None."""
        target_dir = self.output_dir / subfolder if subfolder else self.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        local_path = target_dir / (doc_id + ".pdf")

        if local_path.exists():
            return str(local_path)

        try:
            time.sleep(REQUEST_DELAY_S)
            resp = self.session.get(url, timeout=60, stream=True)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            logger.debug(f"DRDO: downloaded {local_path}")
            return str(local_path)
        except requests.RequestException as e:
            logger.warning(f"DRDO: PDF download failed for {url}: {e}")
            return None
