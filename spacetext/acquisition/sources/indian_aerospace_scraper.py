"""
Indian Aerospace Journals & Private Companies Scraper
=====================================================
Harvests from:
  1. NISCAIR OAI-PMH (Indian Journal of Radio & Space Physics, etc.)
  2. Indian Institute of Space Science and Technology (IIST) publications
  3. Indian private aerospace companies (AgniKul, Skyroot, Bellatrix, Dhruva)
  4. Shodhganga (Indian thesis repository) for aerospace theses
  5. OpenAlex filtered to Indian aerospace institutions

License: Open access content; sources credited.
"""

import re
import time
import logging
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator, Optional
from urllib.parse import urljoin, quote

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

REQUEST_DELAY = 1.5

HEADERS = {
    "User-Agent": "SpaceText-Research-Bot/1.0 (academic; contact: spacetext@research.edu)",
    "Accept": "text/html,application/xhtml+xml,application/xml",
}

# ── Indian Journal OAI-PMH Endpoints ─────────────────────────────────────────
NISCAIR_OAI = "https://nopr.niscpr.res.in/oai/request"
NISCAIR_SETS = [
    "com_123456789_2",   # Indian Journal of Radio & Space Physics
    "com_123456789_10",  # Defence Science Journal
    "com_123456789_5",   # Journal of Scientific and Industrial Research
]

# ── Indian Private Aerospace Companies ────────────────────────────────────────
PRIVATE_COMPANIES = {
    "agnikul": {
        "name": "AgniKul Cosmos",
        "urls": [
            "https://www.agnikul.in/updates",
            "https://www.agnikul.in/news",
        ],
        "arxiv_queries": ["AgniKul Cosmos rocket", "Agnilet engine 3D printed"],
    },
    "skyroot": {
        "name": "Skyroot Aerospace",
        "urls": [
            "https://www.skyroot.in/news",
            "https://www.skyroot.in/updates",
        ],
        "arxiv_queries": ["Skyroot Vikram rocket India", "Indian private launch vehicle"],
    },
    "bellatrix": {
        "name": "Bellatrix Aerospace",
        "urls": [
            "https://www.bellatrix.in/news",
        ],
        "arxiv_queries": ["Bellatrix Aerospace India propulsion", "microwave plasma thruster India"],
    },
    "dhruva": {
        "name": "Dhruva Space",
        "urls": [
            "https://dhruvaspace.com/news",
        ],
        "arxiv_queries": ["Dhruva Space satellite India", "Indian small satellite"],
    },
}

# ── OpenAlex Indian Aerospace Institution ROR IDs ────────────────────────────
INDIAN_INSTITUTION_SEARCHES = [
    "Indian Space Research Organisation",
    "Vikram Sarabhai Space Centre",
    "ISRO Satellite Centre",
    "Space Applications Centre India",
    "Indian Institute of Space Science Technology",
    "National Aerospace Laboratories India",
    "Defence Research Development Organisation",
    "Liquid Propulsion Systems Centre",
    "VSSC Thiruvananthapuram",
]

# ── General arXiv queries for Indian aerospace ────────────────────────────────
INDIAN_ARXIV_QUERIES = [
    "ISRO spacecraft propulsion India",
    "Indian reusable launch vehicle RLV-TD",
    "Chandrayaan lunar orbit",
    "Gaganyaan human spaceflight India",
    "PSLV launch vehicle India",
    "GSLV cryogenic stage India",
    "Aditya L1 solar India",
    "NavIC IRNSS navigation India",
    "Mangalyaan Mars orbit insertion",
    "Indian microsatellite nanosatellite",
    "National Aerospace Laboratories NAL India",
    "Saraswati rocket student India",
]


class IndianAerospaceJournalsScraper:
    """
    Aggregates Indian aerospace content from journals, private companies,
    and Indian institution-filtered academic sources.

    Usage
    -----
    scraper = IndianAerospaceJournalsScraper(output_dir="./raw_data/indian")
    scraper.connect()
    for doc in scraper.stream_documents():
        print(doc["title"])
    """

    def __init__(
        self,
        output_dir: str = "./raw_data/indian",
        download_pdfs: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.download_pdfs = download_pdfs
        self.session: Optional[requests.Session] = None

    def connect(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        logger.info("IndianAerospaceJournalsScraper: session ready.")

    def close(self) -> None:
        if self.session:
            self.session.close()
            self.session = None

    def stream_documents(self) -> Iterator[dict]:
        """Yield all documents from all Indian aerospace sources."""
        if self.session is None:
            raise RuntimeError("Call connect() before stream_documents().")

        logger.info("Indian Journals: starting NISCAIR OAI-PMH harvest...")
        yield from self._harvest_niscair_oai()

        logger.info("Indian Journals: starting private company scrape...")
        yield from self._scrape_private_companies()

        logger.info("Indian Journals: starting OpenAlex Indian institution harvest...")
        yield from self._harvest_openalex_indian()

        logger.info("Indian Journals: starting arXiv Indian aerospace harvest...")
        yield from self._harvest_arxiv_indian()

        logger.info("Indian Journals: starting Shodhganga thesis harvest...")
        yield from self._harvest_shodhganga()

    # ── NISCAIR OAI-PMH (Indian National Science Academy journals) ────────────

    def _harvest_niscair_oai(self) -> Iterator[dict]:
        """Harvest Indian journals via OAI-PMH from NISCAIR."""
        OAI_NS = {
            "oai": "http://www.openarchives.org/OAI/2.0/",
            "dc":  "http://purl.org/dc/elements/1.1/",
            "oai_dc": "http://www.openarchives.org/OAI/2.0/oai_dc/",
        }
        seen = set()
        total = 0

        for set_id in NISCAIR_SETS:
            params = {
                "verb": "ListRecords",
                "metadataPrefix": "oai_dc",
                "set": set_id,
            }
            resumption_token = None
            pages = 0

            # Sets that are generally aerospace/defence related
            # com_123456789_5 is a general science journal, so it must be heavily filtered
            is_general_journal = (set_id == "com_123456789_5")
            
            AEROSPACE_KEYWORDS = {
                "aerospace", "spacecraft", "satellite", "rocket", "propulsion",
                "navigation", "avionics", "ISRO", "DRDO", "launch vehicle",
                "orbit", "reentry", "aerodynamics", "flight", "missile", "radar",
                "aeronautical", "space", "aircraft", "tejas", "uav"
            }

            while pages < 30: # Cap at 30 pages per set (approx 3000 docs max)
                if resumption_token:
                    params = {"verb": "ListRecords", "resumptionToken": resumption_token}

                try:
                    time.sleep(REQUEST_DELAY)
                    resp = self.session.get(NISCAIR_OAI, params=params, timeout=30)
                    resp.raise_for_status()
                except Exception as e:
                    logger.debug(f"NISCAIR OAI failed for set {set_id}: {e}")
                    break

                try:
                    root = ET.fromstring(resp.content)
                except Exception:
                    break

                for record in root.findall(".//oai:record", OAI_NS):
                    identifier = record.findtext(".//oai:identifier", namespaces=OAI_NS) or ""
                    if identifier in seen:
                        continue
                    seen.add(identifier)

                    dc = record.find(".//oai_dc:dc", OAI_NS)
                    if dc is None:
                        continue

                    title = " ".join(t.text or "" for t in dc.findall("dc:title", OAI_NS)).strip()
                    abstract = " ".join(t.text or "" for t in dc.findall("dc:description", OAI_NS)).strip()
                    year = " ".join(t.text or "" for t in dc.findall("dc:date", OAI_NS))[:4]
                    source_url = " ".join(t.text or "" for t in dc.findall("dc:identifier", OAI_NS))

                    if not title:
                        continue

                    # Filter general journals for aerospace relevance
                    if is_general_journal:
                        combined_text = (title + " " + abstract).lower()
                        if not any(kw.lower() in combined_text for kw in AEROSPACE_KEYWORDS):
                            continue

                    # Download PDF if url ends with pdf
                    local_pdf = None
                    pdf_url = None
                    for id_tag in dc.findall("dc:identifier", OAI_NS):
                        if id_tag.text and id_tag.text.lower().endswith(".pdf"):
                            pdf_url = id_tag.text
                            break
                    if self.download_pdfs and pdf_url:
                        doc_id = hashlib.sha256(pdf_url.encode()).hexdigest()[:24]
                        local_pdf = self._download_pdf(pdf_url, doc_id)

                    doc_id = hashlib.sha256(identifier.encode()).hexdigest()[:24]
                    yield {
                        "id": doc_id,
                        "title": title,
                        "abstract": abstract or f"Indian aerospace journal article: {title}",
                        "year": year,
                        "source": "indian_journals",
                        "source_url": source_url,
                        "local_pdf_path": local_pdf,
                    }
                    total += 1

                # Check for resumption token
                rt_elem = root.find(".//oai:resumptionToken", OAI_NS)
                if rt_elem is not None and rt_elem.text:
                    resumption_token = rt_elem.text
                    pages += 1
                else:
                    break

        logger.info(f"NISCAIR OAI: harvested {total} records.")

    # ── Indian Private Aerospace Companies ────────────────────────────────────

    def _scrape_private_companies(self) -> Iterator[dict]:
        """Scrape news/updates from Indian private aerospace companies."""
        seen = set()
        total = 0

        for company_key, company_info in PRIVATE_COMPANIES.items():
            company_name = company_info["name"]

            # Scrape company website
            for url in company_info["urls"]:
                try:
                    time.sleep(REQUEST_DELAY)
                    resp = self.session.get(url, timeout=20)
                    if resp.status_code != 200:
                        continue
                except Exception as e:
                    logger.debug(f"Company {company_name}: failed to fetch {url}: {e}")
                    continue

                soup = BeautifulSoup(resp.text, "html.parser")

                # Extract articles/news items
                items = soup.find_all(["article", "div", "section"],
                                       class_=re.compile(r"(news|blog|post|update|card|item)", re.I))
                for item in items[:30]:
                    h_tag = item.find(["h1", "h2", "h3", "h4", "h5"])
                    if not h_tag:
                        continue
                    title = h_tag.get_text(strip=True)
                    if not title or title in seen:
                        continue
                    seen.add(title)

                    # Get body text
                    paras = item.find_all("p")
                    body = " ".join(p.get_text(strip=True) for p in paras[:4])

                    if not body:
                        body = f"{company_name} aerospace news: {title}. Indian private space company focusing on launch vehicles and satellite services."

                    doc_id = hashlib.sha256(title.encode()).hexdigest()[:24]
                    yield {
                        "id": doc_id,
                        "title": f"[{company_name}] {title}",
                        "abstract": body[:2000],
                        "year": self._extract_year(title + body),
                        "source": "indian_private",
                        "source_url": url,
                        "local_pdf_path": None,
                    }
                    total += 1

            # arXiv queries for this company
            for query in company_info["arxiv_queries"]:
                yield from self._arxiv_query(query, source="indian_private", seen=seen)

        logger.info(f"Private companies: scraped {total} records.")

    # ── OpenAlex Indian Institutions ──────────────────────────────────────────

    def _harvest_openalex_indian(self) -> Iterator[dict]:
        """Query OpenAlex for papers from Indian aerospace institutions."""
        OPENALEX_URL = "https://api.openalex.org/works"
        seen = set()
        total = 0

        for institution_query in INDIAN_INSTITUTION_SEARCHES:
            params = {
                "search": institution_query,
                "filter": "has_abstract:true,is_oa:true",
                "per-page": 50,
                "cursor": "*",
            }
            pages = 0
            while pages < 4:
                try:
                    time.sleep(1.0)
                    resp = self.session.get(OPENALEX_URL, params=params, timeout=30)
                    resp.raise_for_status()
                except Exception as e:
                    logger.debug(f"OpenAlex Indian query failed: {e}")
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
                        "source": "indian_journals",
                        "source_url": work.get("id", ""),
                        "local_pdf_path": None,
                    }
                    total += 1

                next_cursor = data.get("meta", {}).get("next_cursor")
                if next_cursor:
                    params["cursor"] = next_cursor
                    pages += 1
                else:
                    break

        logger.info(f"OpenAlex Indian institutions: harvested {total} records.")

    # ── arXiv Indian Aerospace Papers ─────────────────────────────────────────

    def _harvest_arxiv_indian(self) -> Iterator[dict]:
        """Query arXiv for Indian aerospace papers."""
        seen = set()
        for query in INDIAN_ARXIV_QUERIES:
            yield from self._arxiv_query(query, source="isro", seen=seen)
        logger.info(f"arXiv Indian aerospace: harvested {len(seen)} papers.")

    # ── Shodhganga Indian Thesis Repository ───────────────────────────────────

    def _harvest_shodhganga(self) -> Iterator[dict]:
        """
        Harvest aerospace theses from Shodhganga (India's national thesis repository).
        Uses OAI-PMH endpoint.
        """
        SHODHGANGA_OAI = "https://shodhganga.inflibnet.ac.in/oai/request"
        OAI_NS = {
            "oai": "http://www.openarchives.org/OAI/2.0/",
            "dc": "http://purl.org/dc/elements/1.1/",
            "oai_dc": "http://www.openarchives.org/OAI/2.0/oai_dc/",
        }
        # Shodhganga set IDs for aerospace/space/defence subjects
        SHODHGANGA_SETS = [
            "hdl_10603_3",   # Engineering
        ]
        AEROSPACE_KEYWORDS = {
            "aerospace", "spacecraft", "satellite", "rocket", "propulsion",
            "navigation", "avionics", "ISRO", "DRDO", "launch vehicle",
            "orbit", "reentry", "aerodynamics", "flight", "missile",
        }
        seen = set()
        total = 0

        for set_id in SHODHGANGA_SETS:
            params = {
                "verb": "ListRecords",
                "metadataPrefix": "oai_dc",
                "set": set_id,
            }
            resumption_token = None
            pages = 0

            while pages < 5:
                if resumption_token:
                    params = {"verb": "ListRecords", "resumptionToken": resumption_token}

                try:
                    time.sleep(REQUEST_DELAY)
                    resp = self.session.get(SHODHGANGA_OAI, params=params, timeout=30)
                    resp.raise_for_status()
                except Exception as e:
                    logger.debug(f"Shodhganga OAI failed: {e}")
                    break

                try:
                    root = ET.fromstring(resp.content)
                except Exception:
                    break

                for record in root.findall(".//oai:record", OAI_NS):
                    identifier = record.findtext(".//oai:identifier", namespaces=OAI_NS) or ""
                    if identifier in seen:
                        continue

                    dc = record.find(".//oai_dc:dc", OAI_NS)
                    if dc is None:
                        continue

                    title = " ".join(t.text or "" for t in dc.findall("dc:title", OAI_NS)).strip()
                    abstract = " ".join(t.text or "" for t in dc.findall("dc:description", OAI_NS)).strip()

                    # Only keep aerospace-relevant theses
                    combined_text = (title + " " + abstract).lower()
                    if not any(kw.lower() in combined_text for kw in AEROSPACE_KEYWORDS):
                        continue

                    seen.add(identifier)
                    year = " ".join(t.text or "" for t in dc.findall("dc:date", OAI_NS))[:4]
                    doc_id = hashlib.sha256(identifier.encode()).hexdigest()[:24]

                    yield {
                        "id": doc_id,
                        "title": f"[Thesis] {title}",
                        "abstract": abstract or f"Indian aerospace research thesis: {title}",
                        "year": year,
                        "source": "indian_journals",
                        "source_url": identifier,
                        "local_pdf_path": None,
                    }
                    total += 1

                rt_elem = root.find(".//oai:resumptionToken", OAI_NS)
                if rt_elem is not None and rt_elem.text:
                    resumption_token = rt_elem.text
                    pages += 1
                else:
                    break

        logger.info(f"Shodhganga: harvested {total} aerospace theses.")

    # ── Shared Helpers ────────────────────────────────────────────────────────

    def _arxiv_query(self, query: str, source: str, seen: set) -> Iterator[dict]:
        """Query arXiv Atom feed and yield documents."""
        atom_url = (
            f"http://export.arxiv.org/api/query?"
            f"search_query=all:{quote(query)}&start=0&max_results=50"
        )
        try:
            time.sleep(REQUEST_DELAY)
            resp = self.session.get(atom_url, timeout=20)
            resp.raise_for_status()
        except Exception as e:
            logger.debug(f"arXiv query failed for '{query}': {e}")
            return

        try:
            root = ET.fromstring(resp.content)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            entries = root.findall("atom:entry", ns)
        except Exception:
            return

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
                "source": source,
                "source_url": f"https://arxiv.org/abs/{arxiv_id}",
                "local_pdf_path": None,
            }

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
            logger.debug(f"Indian journals: PDF download failed for {url}: {e}")
            return None
