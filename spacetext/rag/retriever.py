"""
SpaceText GarudDB Retriever
============================
Connects to the GarudDB aerospace index and provides:
  - retrieve()          : flat semantic search (top_k docs) for Q&A
  - retrieve_cluster()  : taxonomy-aware cluster retrieval for summarization
"""

import logging
from typing import List, Tuple, Optional

try:
    from garuddb import PersistentClient
    from garuddb.embeddings import GarudEmbedV5Function, DefaultEmbeddingFunction
    GARUDDB_AVAILABLE = True
except ImportError:
    GARUDDB_AVAILABLE = False

logger = logging.getLogger("spacetext.rag.retriever")


class GarudDBRetriever:
    """
    Singleton-per-path retriever for the GarudDB aerospace index.

    FIX FLAW 6: Prevents double-lock errors by reusing an existing client
    for the same database path.
    FIX FLAW 7: retrieve_cluster() fetches 20+ docs for summarization,
    not just 3 for Q&A.
    """

    # Class-level singleton cache keyed by db_path
    _instances: dict = {}

    def __new__(cls, db_path: str = "./aerospace_taxonomy.adk"):
        """Return a cached instance for the given db_path to avoid double-lock."""
        resolved = str(db_path)
        if resolved not in cls._instances:
            instance = super().__new__(cls)
            instance._initialized = False
            cls._instances[resolved] = instance
        return cls._instances[resolved]

    def __init__(self, db_path: str = "./aerospace_taxonomy.adk"):
        if self._initialized:
            return
        self.db_path = db_path
        self.collection = None
        self._init_db()
        self._initialized = True

    def _init_db(self):
        if not GARUDDB_AVAILABLE:
            raise RuntimeError("GarudDB is not installed.")

        client = PersistentClient(path=self.db_path)
        existing = client.list_collections()
        if "aerospace_taxonomy" not in existing:
            raise ValueError(f"Collection 'aerospace_taxonomy' not found in {self.db_path}")
        self.collection = client.get_collection("aerospace_taxonomy")
        logger.info(f"GarudDBRetriever: connected ({self.collection.count()} docs)")

    # ── Flat Semantic Search (for Q&A) ─────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieves top_k documents related to the query using semantic search.
        Returns a list of formatted strings: '[Source | Category]\ndoc_text'
        """
        if not self.collection:
            return []

        results = self.collection.query(query_texts=[query], n_results=top_k)

        documents = []
        if results and results.get("documents") and len(results["documents"]) > 0:
            for idx, doc in enumerate(results["documents"][0]):
                if doc is None:
                    continue
                meta = results["metadatas"][0][idx] if results.get("metadatas") else {}
                source = meta.get("source", "Unknown")
                label = meta.get("taxonomy_label", "Unknown")
                documents.append(f"[Source: {source} | Category: {label}]\n{doc}")

        return documents

    # ── Taxonomy Cluster Retrieval (for Summarization) ─────────────────────────

    def retrieve_cluster(
        self,
        topic: str,
        top_k: int = 20,
        source_filter: Optional[str] = None,
    ) -> List[Tuple[str, str]]:
        """
        Retrieves a taxonomy-aware cluster of documents for multi-doc summarization.

        Returns a list of (source, doc_text) tuples.
        Uses GarudDB's hyperbolic find_subtree() if the collection is hyperbolic,
        otherwise falls back to semantic search with a larger top_k.

        Args:
            topic:         The topic to search for (e.g., "DRDO inertial navigation")
            top_k:         Number of documents to retrieve (default 20 for summarization)
            source_filter: Optional filter by source (e.g., "drdo", "isro", "nasa")
        """
        if not self.collection:
            return []

        try:
            where = {"source": source_filter} if source_filter else None
            # The GarudDB query() method automatically uses hyperbolic search
            # (search_topology) if the index was built with is_hyperbolic=True
            results = self.collection.query(
                query_texts=[topic],
                n_results=top_k,
                where=where,
            )
        except Exception as e:
            logger.warning(f"Cluster retrieval failed: {e}")
            return []

        cluster = []
        docs_list = results.get("documents", [[]])[0] if isinstance(results.get("documents", []), list) else results.get("documents", [])
        metas_list = results.get("metadatas", [[]])[0] if isinstance(results.get("metadatas", []), list) else results.get("metadatas", [])
        distances_list = results.get("distances", [[]])[0] if isinstance(results.get("distances", []), list) else results.get("distances", [])

        # DYNAMIC FILTERING: Drop documents that are not semantically relevant
        valid_indices = []
        if distances_list and len(distances_list) > 0:
            best_dist = distances_list[0]
            # Since distances are typically 0 to 2, we use a much tighter mathematical threshold
            threshold = best_dist + 0.15 
            for idx, dist in enumerate(distances_list):
                # Always keep at least the top 3, otherwise drop if distance is too large
                if idx < 3 or dist <= threshold:
                    valid_indices.append(idx)
        else:
            valid_indices = list(range(len(docs_list)))

        for idx in valid_indices:
            if idx >= len(docs_list) or docs_list[idx] is None:
                continue
            meta = metas_list[idx] if idx < len(metas_list) else {}
            source = meta.get("source", "unknown")
            if source_filter and source != source_filter:
                continue
            cluster.append((source, docs_list[idx]))

        logger.info(f"Retrieved {len(cluster)} docs for topic: '{topic}'")
        return cluster
