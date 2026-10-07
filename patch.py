import re

with open("spacetext/acquisition/pipeline.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. Imports
code = code.replace("from typing import Iterator, List, Optional", "from typing import Iterator, List, Optional\nimport concurrent.futures\nimport requests\nimport hashlib")

# 2. _get_source_stream download_pdfs=False
code = code.replace("download_pdfs=True", "download_pdfs=False")

# 3. Add _download_pdf and _add_to_garuddb_batch
helper_code = """
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
            doc_text = f"{r['title']}. {r['abstract'][:500]}"
            if not doc_text.strip(): continue
            docs.append(doc_text)
            metas.append({
                "taxonomy_label": r["taxonomy_label"][:64],
                "depth": r["taxonomy_depth"],
                "source": r["source"],
                "year": r.get("year", "")[:4],
                "path_len": len(r["taxonomy_path"]),
            })
            ids.append(r["id"])
        
        if docs:
            self.collection.add(documents=docs, metadatas=metas, ids=ids)

"""
code = code.replace("    def _process_document(", helper_code + "    def _process_document(")

# 4. Replace run() method entirely
run_start = code.find("    def run(self, sources: List[str] = None):")
run_end = code.find("    # ── Source Routing ────────────────────────────────────────────────────────")
if run_start != -1 and run_end != -1:
    new_run = """    def run(self, sources: List[str] = None):
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
                            
                            corpus_file.write(json.dumps(record, ensure_ascii=False) + "\\n")
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
                            
                        corpus_file.write(json.dumps(record, ensure_ascii=False) + "\\n")
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
"""
    code = code[:run_start] + new_run + "\n" + code[run_end:]

with open("spacetext/acquisition/pipeline.py", "w", encoding="utf-8") as f:
    f.write(code)
