"""Hybrid retrieval for the existing GeM Chroma embedding collection.

The dense index remains in Chroma.  This module fuses it with Chroma's
already-persisted FTS5/BM25 document index using reciprocal-rank fusion
(RRF). This gives exact product names, standards, bid IDs and numbers a
chance to win without losing semantic recall.  A separate sidecar index is
still available as a compatibility fallback, but is not needed for the
current collection.

Typical use
-----------
Search it::

    python gem_hybrid_retrieval.py --query "bullet resistant vehicle warranty"

The script does not alter ``chroma_db``.  ``--build-lexical-index`` is only
needed to create an optional compatibility sidecar at
``chroma_db/gem_bid_chunks_lexical.sqlite3``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


DEFAULT_CHROMA_PATH = Path("chroma_db")
DEFAULT_COLLECTION = "gem_bid_chunks"
DEFAULT_LEXICAL_DB = "gem_bid_chunks_lexical.sqlite3"

# These words are useful for natural-language phrasing but are not evidence
# that a bid matches a procurement request on their own.
GENERIC_QUERY_TERMS = frozenset({
    "a", "an", "and", "as", "at", "by", "clearly", "details", "for", "from", "generic",
    "goods", "in", "item", "items", "not", "of", "on", "or", "per", "product", "products",
    "requirement", "required", "service", "services", "specified", "the", "to", "use", "with",
    "category",
})


@dataclass(frozen=True)
class SearchResult:
    """A chunk returned by :class:`HybridRetriever`."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fts_query(query: str) -> str:
    """Turn free text into a safe FTS5 AND query.

    Quoting individual terms prevents punctuation such as ``GEM/2024/B``
    from becoming FTS syntax.  Very short tokens are ignored because they
    tend to create noise in procurement text ("of", "to", etc.).
    """
    terms = re.findall(r"[\w.-]+", query, flags=re.UNICODE)
    terms = [term.replace('"', '""') for term in terms if len(term) > 1]
    # OR retains recall for conversational queries whose wording differs from
    # a bid title (e.g. "for police force" vs "required by police"). BM25
    # still ranks records containing more of the query terms above others.
    return " OR ".join(f'"{term}"' for term in terms)


def _matches_where(metadata: Mapping[str, Any], where: Mapping[str, Any] | None) -> bool:
    """Apply the useful Chroma-style filter subset to lexical candidates."""
    if not where:
        return True
    if "$and" in where:
        return all(_matches_where(metadata, item) for item in where["$and"])
    if "$or" in where:
        return any(_matches_where(metadata, item) for item in where["$or"])

    for key, expected in where.items():
        value = metadata.get(key)
        if isinstance(expected, Mapping):
            for operator, operand in expected.items():
                if operator == "$eq" and value != operand:
                    return False
                if operator == "$ne" and value == operand:
                    return False
                if operator == "$in" and value not in operand:
                    return False
                if operator == "$nin" and value in operand:
                    return False
                if operator == "$gt" and not (value is not None and value > operand):
                    return False
                if operator == "$gte" and not (value is not None and value >= operand):
                    return False
                if operator == "$lt" and not (value is not None and value < operand):
                    return False
                if operator == "$lte" and not (value is not None and value <= operand):
                    return False
        elif value != expected:
            return False
    return True


class LexicalIndex:
    """Persistent BM25 index for documents already stored in Chroma.

    The index stores metadata as JSON so it can apply the same common filters
    as the dense side.  It is intentionally separate from Chroma's SQLite
    database: Chroma owns that file and may migrate its schema.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._df_cache: dict[str, int] = {}
        self._total_docs_cache: int | None = None

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialise(self) -> None:
        with closing(self._connect()) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5("
                "chunk_id UNINDEXED, title, text, tokenize='unicode61 remove_diacritics 2')"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS chunks_metadata ("
                "chunk_id TEXT PRIMARY KEY, metadata_json TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS index_state (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            db.commit()

    def term_document_frequency(self, term: str) -> int:
        """Count of chunks whose text contains this term, memoized per-process."""
        cache = getattr(self, "_df_cache", None)
        if cache is None:
            cache = self._df_cache = {}
        key = term.lower()
        if key in cache:
            return cache[key]
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT COUNT(*) AS c FROM chunks_fts WHERE chunks_fts MATCH ?",
                (f'"{key}"',),
            ).fetchone()
        df = row["c"] if row else 0
        cache[key] = df
        return df

    def total_document_count(self) -> int:
        total = getattr(self, "_total_docs_cache", None)
        if total is None:
            with closing(self._connect()) as db:
                row = db.execute("SELECT COUNT(*) AS c FROM chunks_metadata").fetchone()
            total = self._total_docs_cache = (row["c"] if row else 0)
        return total

    def clear(self) -> None:
        """Reset a full-build sidecar without scanning every FTS row.

        ``DELETE FROM`` on a large FTS5 table is much slower than recreating
        this disposable index.  Only the sidecar tables are dropped; Chroma's
        persistent database is a different file and is never touched.
        """
        with closing(self._connect()) as db:
            db.execute("DROP TABLE IF EXISTS chunks_fts")
            db.execute("DROP TABLE IF EXISTS chunks_metadata")
            db.execute("DROP TABLE IF EXISTS index_state")
            db.commit()
        self._df_cache.clear()
        self._total_docs_cache = None
        self.initialise()

    def add(
        self,
        ids: Sequence[str],
        documents: Sequence[str | None],
        metadatas: Sequence[Mapping[str, Any] | None],
        *,
        replace: bool = True,
    ) -> None:
        if not (len(ids) == len(documents) == len(metadatas)):
            raise ValueError("ids, documents, and metadatas must have equal lengths")
        self.initialise()
        records = [
            (
                chunk_id,
                str((metadata or {}).get("bid_title") or ""),
                str(document or ""),
                json.dumps(dict(metadata or {}), ensure_ascii=False, separators=(",", ":")),
            )
            for chunk_id, document, metadata in zip(ids, documents, metadatas)
        ]
        with closing(self._connect()) as db:
            if replace:
                # FTS5 tables have no primary key. Refreshing a small subset
                # therefore removes its prior rows before reinserting them.
                db.executemany("DELETE FROM chunks_fts WHERE chunk_id = ?", [(row[0],) for row in records])
                db.executemany("DELETE FROM chunks_metadata WHERE chunk_id = ?", [(row[0],) for row in records])
            db.executemany("INSERT INTO chunks_fts(chunk_id, title, text) VALUES (?, ?, ?)", [row[:3] for row in records])
            db.executemany(
                "INSERT INTO chunks_metadata(chunk_id, metadata_json) VALUES (?, ?) "
                "ON CONFLICT(chunk_id) DO UPDATE SET metadata_json=excluded.metadata_json",
                [(row[0], row[3]) for row in records],
            )
            db.commit()

    def set_state(self, **values: Any) -> None:
        self.initialise()
        with closing(self._connect()) as db:
            db.executemany(
                "INSERT INTO index_state(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                [(key, json.dumps(value)) for key, value in values.items()],
            )
            db.commit()

    def state(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        with closing(self._connect()) as db:
            try:
                return {row["key"]: json.loads(row["value"]) for row in db.execute("SELECT key, value FROM index_state")}
            except sqlite3.OperationalError:
                return {}

    def search(self, query: str, limit: int = 50, where: Mapping[str, Any] | None = None) -> list[SearchResult]:
        """Return BM25-ranked candidates, post-filtered using their metadata."""
        if not self.path.exists():
            raise FileNotFoundError(f"Lexical index not found: {self.path}. Run --build-lexical-index first.")
        fts_query = _fts_query(query)
        if not fts_query:
            return []

        # Filters happen locally after BM25. Request a sizeable candidate pool
        # so common ministry/organisation filters still produce enough hits.
        fetch_limit = max(limit * 25, 250)
        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT f.chunk_id, f.text, m.metadata_json, "
                "bm25(chunks_fts, 0.0, 3.0, 1.0) AS bm25_score "
                "FROM chunks_fts f JOIN chunks_metadata m USING(chunk_id) "
                "WHERE chunks_fts MATCH ? ORDER BY bm25_score LIMIT ?",
                (fts_query, fetch_limit),
            ).fetchall()

        results: list[SearchResult] = []
        for row in rows:
            metadata = json.loads(row["metadata_json"])
            if _matches_where(metadata, where):
                results.append(SearchResult(row["chunk_id"], row["text"], metadata, float(row["bm25_score"])))
                if len(results) == limit:
                    break
        return results


def reciprocal_rank_fusion(
    dense_results: Sequence[SearchResult],
    lexical_results: Sequence[SearchResult],
    *,
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    lexical_weight: float = 1.0,
) -> list[SearchResult]:
    """Fuse ranked results without comparing incompatible score scales."""
    combined: dict[str, dict[str, Any]] = {}
    for source, results, weight in (("dense", dense_results, dense_weight), ("lexical", lexical_results, lexical_weight)):
        for rank, result in enumerate(results, start=1):
            item = combined.setdefault(
                result.chunk_id,
                {"result": result, "score": 0.0, "dense_rank": None, "lexical_rank": None},
            )
            item["score"] += weight / (rrf_k + rank)
            item[f"{source}_rank"] = rank

    return [
        SearchResult(
            chunk_id=item["result"].chunk_id,
            text=item["result"].text,
            metadata=item["result"].metadata,
            score=item["score"],
            dense_rank=item["dense_rank"],
            lexical_rank=item["lexical_rank"],
        )
        for item in sorted(combined.values(), key=lambda value: value["score"], reverse=True)
    ]


def _query_terms(query: str) -> list[str]:
    """Return meaningful terms while retaining short technical acronyms."""
    raw_terms = re.findall(r"[A-Za-z0-9]+", query)
    return [
        term.lower()
        for term in raw_terms
        if term.lower() not in GENERIC_QUERY_TERMS and (len(term) >= 3 or term.isupper())
    ]


def _normalise_for_match(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def lexical_match_quality(
    query: str,
    result: SearchResult,
    *,
    term_df: Callable[[str], int] | None = None,
    total_docs: int = 1,
) -> float:
    """Estimate whether a lexical candidate actually supports the query using term IDF weighting."""
    if re.fullmatch(r"GEM/\d{4}/[BR]/\d+", query.strip(), flags=re.IGNORECASE):
        return 1.0 if str(result.metadata.get("bid_id", "")).lower() == query.strip().lower() else 0.0

    terms = _query_terms(query)
    if not terms:
        return 0.0

    text = _normalise_for_match(result.text)
    title = _normalise_for_match(str(result.metadata.get("bid_title") or ""))
    phrase = _normalise_for_match(query)
    if phrase and phrase in text:
        return 1.0

    def idf(term: str) -> float:
        if term_df is None:
            return 1.0  # falls back to old flat-count behavior
        df = term_df(term)
        return math.log((total_docs + 1) / (df + 1)) + 1.0

    weights = {term: idf(term) for term in terms}
    total_weight = sum(weights.values()) or 1.0

    text_weight = sum(w for t, w in weights.items() if re.search(rf"\b{re.escape(t)}\b", text))
    title_weight = sum(w for t, w in weights.items() if re.search(rf"\b{re.escape(t)}\b", title))

    text_coverage = text_weight / total_weight
    title_coverage = title_weight / total_weight

    # Same guard as before, but weight-aware: a query dominated by rare terms
    # that never appear anywhere is no longer "rescued" by common-term matches.
    if len(terms) >= 3 and text_weight < 0.5 * total_weight and title_weight == 0:
        return 0.0

    if title_coverage >= 0.5:
        top_term = max(weights, key=weights.get)
        if re.search(rf"\b{re.escape(top_term)}\b", title):
            return max(0.85, title_coverage)
        return title_coverage
    return max(title_coverage, text_coverage * 0.70)


def dense_confidence(distance: float) -> float:
    """Map this collection's L2 distances to a conservative confidence.

    Values are calibrated from the supplied evaluation set: specific good
    matches cluster near 1.40--1.55, while unrelated nearest-neighbours are
    commonly 1.65 or above. Keep this deliberately simple and easy to tune.
    """
    if distance <= 1.45:
        return 1.0
    if distance <= 1.55:
        return 0.85
    if distance <= 1.65:
        return 0.55
    if distance <= 1.75:
        return 0.20
    return 0.0


def confidence_aware_fusion(
    query: str,
    dense_results: Sequence[SearchResult],
    lexical_results: Sequence[SearchResult],
    *,
    term_df: Callable[[str], int] | None = None,
    total_docs: int = 1,
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    lexical_weight: float = 1.0,
) -> list[SearchResult]:
    """Fuse rankings while accounting for actual lexical/dense confidence."""
    strong_lexical = [
        (result, lexical_match_quality(query, result, term_df=term_df, total_docs=total_docs))
        for result in lexical_results
    ]
    strong_lexical = [(result, quality) for result, quality in strong_lexical if quality >= 0.20]
    # Do not present a nearest vector neighbour as an answer if the query has
    # neither lexical support nor a sufficiently close semantic candidate.
    if not strong_lexical and not any(dense_confidence(result.score) >= 0.55 for result in dense_results):
        return []

    combined: dict[str, dict[str, Any]] = {}
    for rank, result in enumerate(dense_results, start=1):
        confidence = dense_confidence(result.score)
        if confidence == 0:
            continue
        item = combined.setdefault(
            result.chunk_id,
            {"result": result, "score": 0.0, "dense_rank": None, "lexical_rank": None},
        )
        item["score"] += dense_weight * confidence / (rrf_k + rank)
        item["dense_rank"] = rank

    for rank, (result, quality) in enumerate(strong_lexical, start=1):
        # Title/phrase matches dominate weak dense neighbours; a partial
        # lexical match still complements a genuinely close vector result.
        confidence = 0.50 + 3.50 * quality
        item = combined.setdefault(
            result.chunk_id,
            {"result": result, "score": 0.0, "dense_rank": None, "lexical_rank": None},
        )
        item["score"] += lexical_weight * confidence / (rrf_k + rank)
        item["lexical_rank"] = rank

    return [
        SearchResult(
            chunk_id=item["result"].chunk_id,
            text=item["result"].text,
            metadata=item["result"].metadata,
            score=item["score"],
            dense_rank=item["dense_rank"],
            lexical_rank=item["lexical_rank"],
        )
        for item in sorted(combined.values(), key=lambda value: value["score"], reverse=True)
    ]


BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class HybridRetriever:
    def __init__(
        self,
        chroma_path: str | Path = DEFAULT_CHROMA_PATH,
        collection_name: str = DEFAULT_COLLECTION,
        lexical_db_path: str | Path | None = None,
        *,
        query_model_name: str = "BAAI/bge-small-en-v1.5",
    ):
        self.chroma_path = Path(chroma_path)
        self.collection_name = collection_name
        self.lexical = LexicalIndex(lexical_db_path or self.chroma_path / f"{collection_name}_lexical.sqlite3")
        self.query_model_name = query_model_name
        self._query_model: Any | None = None  # lazy-loaded SentenceTransformer
        self._collection: Any | None = None

    @property
    def collection(self) -> Any:
        if self._collection is None:
            try:
                import chromadb
            except ImportError as error:
                raise RuntimeError("chromadb is required for dense retrieval.") from error
            client = chromadb.PersistentClient(path=str(self.chroma_path))
            # Deliberately no embedding_function here. The corpus vectors were
            # inserted via collection.add(embeddings=...) with precomputed BGE
            # vectors, so Chroma never actually embedded anything itself -- but
            # it still persisted a nominal "default" embedding function in the
            # collection config since none was specified at creation time.
            # Passing any embedding_function into get_collection() now trips
            # Chroma's conflict check against that (misleading) persisted
            # config. We avoid the whole problem by always supplying
            # query_embeddings ourselves (see _embed_query) and never asking
            # Chroma to embed anything.
            self._collection = client.get_collection(self.collection_name)
        return self._collection

    def _embed_query(self, query: str) -> list[float]:
        if self._query_model is None:
            from sentence_transformers import SentenceTransformer
            self._query_model = SentenceTransformer(self.query_model_name)
        vector = self._query_model.encode(
            [BGE_QUERY_PREFIX + query],
            normalize_embeddings=True,   # matches how the corpus was embedded
            convert_to_numpy=True,
        )[0]
        return vector.tolist()

    def _build_from_chroma_sqlite(self, total: int, batch_size: int) -> int:
        """Build from Chroma's local metadata segment without loading it all."""
        source = self.chroma_path / "chroma.sqlite3"
        if not source.exists():
            raise FileNotFoundError(source)

        uri = f"file:{source.resolve().as_posix()}?mode=ro"
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        def flush() -> None:
            if ids:
                self.lexical.add(ids, documents, metadatas, replace=False)
                print(f"Indexed {len_indexed[0]:,}/{total:,} chunks", flush=True)
                ids.clear()
                documents.clear()
                metadatas.clear()

        len_indexed = [0]
        with closing(sqlite3.connect(uri, uri=True)) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(
                "SELECT e.embedding_id, m.key, m.string_value, m.int_value, m.float_value, m.bool_value "
                "FROM embeddings e "
                "JOIN segments s ON s.id = e.segment_id "
                "JOIN collections c ON c.id = s.collection "
                "JOIN embedding_metadata m ON m.id = e.id "
                "WHERE c.name = ? AND s.scope = 'METADATA' "
                "ORDER BY e.id, m.key",
                (self.collection_name,),
            )
            current_id: str | None = None
            current_document = ""
            current_metadata: dict[str, Any] = {}

            def finish_current() -> None:
                nonlocal current_id, current_document, current_metadata
                if current_id is None:
                    return
                ids.append(current_id)
                documents.append(current_document)
                metadatas.append(current_metadata)
                len_indexed[0] += 1
                if len(ids) >= batch_size:
                    flush()

            for row in rows:
                chunk_id = row["embedding_id"]
                if current_id != chunk_id:
                    finish_current()
                    current_id = chunk_id
                    current_document = ""
                    current_metadata = {}
                if row["string_value"] is not None:
                    value: Any = row["string_value"]
                elif row["int_value"] is not None:
                    value = row["int_value"]
                elif row["float_value"] is not None:
                    value = row["float_value"]
                elif row["bool_value"] is not None:
                    value = bool(row["bool_value"])
                else:
                    continue
                if row["key"] == "chroma:document":
                    current_document = str(value)
                else:
                    current_metadata[row["key"]] = value
            finish_current()
        flush()
        return len_indexed[0]

    def build_lexical_index(self, *, batch_size: int = 1_000, rebuild: bool = False) -> int:
        """Copy Chroma documents/metadata into the BM25 sidecar in batches."""
        total = self.collection.count()
        state = self.lexical.state()
        if not rebuild and state.get("collection") == self.collection_name and state.get("chunk_count") == total:
            return 0

        self.lexical.clear()
        try:
            indexed = self._build_from_chroma_sqlite(total, batch_size)
        except (sqlite3.Error, FileNotFoundError) as error:
            print(f"Direct Chroma metadata scan unavailable ({error}); using Chroma API.", flush=True)
            indexed = 0
            for offset in range(0, total, batch_size):
                batch = self.collection.get(
                    limit=batch_size,
                    offset=offset,
                    include=["documents", "metadatas"],
                )
                self.lexical.add(batch["ids"], batch["documents"], batch["metadatas"], replace=False)
                indexed += len(batch["ids"])
                print(f"Indexed {indexed:,}/{total:,} chunks", flush=True)
        self.lexical.set_state(collection=self.collection_name, chunk_count=indexed, indexed_at=_utcnow())
        return indexed

    def _dense_search(
            self,
            query: str,
            *,
            candidates: int,
            where: Mapping[str, Any] | None,
            query_embedding: Sequence[float] | None,
    ) -> list[SearchResult]:
        if query_embedding is None:
            query_embedding = self._embed_query(query)
        arguments: dict[str, Any] = {
            "n_results": candidates,
            "where": where,
            "include": ["documents", "metadatas", "distances"],
            "query_embeddings": [list(query_embedding)],
        }
        response = self.collection.query(**arguments)
        return [
            SearchResult(chunk_id, document or "", dict(metadata or {}), float(distance))
            for chunk_id, document, metadata, distance in zip(
                response["ids"][0], response["documents"][0], response["metadatas"][0], response["distances"][0]
            )
        ]

    def _native_term_document_frequency(self, term: str) -> int:
        cache = getattr(self, "_native_df_cache", None)
        if cache is None:
            cache = self._native_df_cache = {}
        key = term.lower()
        if key in cache:
            return cache[key]
        source = self.chroma_path / "chroma.sqlite3"
        uri = f"file:{source.resolve().as_posix()}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as db:
            row = db.execute(
                "SELECT COUNT(*) AS c FROM embedding_fulltext_search WHERE embedding_fulltext_search MATCH ?",
                (f'"{key}"',),
            ).fetchone()
        df = row[0] if row else 0
        cache[key] = df
        return df

    def _native_lexical_search(
        self, query: str, limit: int, where: Mapping[str, Any] | None
    ) -> list[SearchResult]:
        """Use Chroma's already-persisted FTS5 document index."""
        source = self.chroma_path / "chroma.sqlite3"
        fts_query = _fts_query(query)
        fetch_limit = max(limit * 25, 250)
        uri = f"file:{source.resolve().as_posix()}?mode=ro"
        try:
            with closing(sqlite3.connect(uri, uri=True)) as db:
                db.row_factory = sqlite3.Row
                candidates: dict[int, sqlite3.Row] = {}
                if fts_query:
                    rows = db.execute(
                        "SELECT f.rowid AS metadata_id, e.embedding_id, f.string_value AS text, "
                        "bm25(embedding_fulltext_search) AS bm25_score "
                        "FROM embedding_fulltext_search f "
                        "JOIN embeddings e ON e.id = f.rowid "
                        "JOIN segments s ON s.id = e.segment_id "
                        "JOIN collections c ON c.id = s.collection "
                        "WHERE embedding_fulltext_search MATCH ? AND c.name = ? AND s.scope = 'METADATA' "
                        "ORDER BY bm25_score LIMIT ?",
                        (fts_query, self.collection_name, fetch_limit),
                    ).fetchall()
                    candidates.update({row["metadata_id"]: row for row in rows})

                if re.fullmatch(r"GEM/\d{4}/[BR]/\d+", query.strip(), flags=re.IGNORECASE):
                    exact_rows = db.execute(
                        "SELECT e.id AS metadata_id, e.embedding_id, d.string_value AS text, -1.0e12 AS bm25_score "
                        "FROM embedding_metadata bid "
                        "JOIN embeddings e ON e.id = bid.id "
                        "JOIN embedding_metadata d ON d.id = e.id AND d.key = 'chroma:document' "
                        "JOIN segments s ON s.id = e.segment_id "
                        "JOIN collections c ON c.id = s.collection "
                        "WHERE bid.key = 'bid_id' AND lower(bid.string_value) = lower(?) "
                        "AND c.name = ? AND s.scope = 'METADATA'",
                        (query.strip(), self.collection_name),
                    ).fetchall()
                    candidates.update({row["metadata_id"]: row for row in exact_rows})

                if not candidates:
                    return []
                placeholders = ",".join("?" for _ in candidates)
                metadata_rows = db.execute(
                    "SELECT id, key, string_value, int_value, float_value, bool_value "
                    f"FROM embedding_metadata WHERE id IN ({placeholders})",
                    list(candidates),
                ).fetchall()
        except sqlite3.Error as error:
            raise RuntimeError(f"Could not query Chroma's native lexical index: {error}") from error

        metadata_by_id: dict[int, dict[str, Any]] = {identifier: {} for identifier in candidates}
        for row in metadata_rows:
            if row["key"] == "chroma:document":
                continue
            value: Any
            if row["string_value"] is not None:
                value = row["string_value"]
            elif row["int_value"] is not None:
                value = row["int_value"]
            elif row["float_value"] is not None:
                value = row["float_value"]
            elif row["bool_value"] is not None:
                value = bool(row["bool_value"])
            else:
                continue
            metadata_by_id[row["id"]][row["key"]] = value

        results = []
        for rank, row in enumerate(sorted(candidates.values(), key=lambda item: item["bm25_score"]), start=1):
            metadata = metadata_by_id[row["metadata_id"]]
            if _matches_where(metadata, where):
                results.append(SearchResult(row["embedding_id"], row["text"] or "", metadata, float(row["bm25_score"])))
                if len(results) == limit:
                    break
        return results

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        where: Mapping[str, Any] | None = None,
        query_embedding: Sequence[float] | None = None,
        dense_candidates: int | None = None,
        lexical_candidates: int | None = None,
        exclude_boilerplate: bool = True,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        lexical_weight: float = 1.0,
    ) -> list[SearchResult]:
        """Return hybrid-ranked chunks."""
        if not query.strip():
            return []
        candidates = max(limit * 4, 40)
        dense = self._dense_search(
            query,
            candidates=dense_candidates or candidates,
            where=where,
            query_embedding=query_embedding,
        )
        sidecar_is_ready = self.lexical.state().get("collection") == self.collection_name
        lexical = (
            self.lexical.search(query, lexical_candidates or candidates, where)
            if sidecar_is_ready
            else self._native_lexical_search(query, lexical_candidates or candidates, where)
        )
        if exclude_boilerplate:
            dense = [result for result in dense if result.metadata.get("is_boilerplate") is not True]
            lexical = [result for result in lexical if result.metadata.get("is_boilerplate") is not True]

        if sidecar_is_ready:
            term_df_fn = self.lexical.term_document_frequency
            total_docs = self.lexical.total_document_count()
        else:
            term_df_fn = self._native_term_document_frequency
            total_docs = self.collection.count()

        return confidence_aware_fusion(
            query,
            dense,
            lexical,
            term_df=term_df_fn,
            total_docs=total_docs,
            rrf_k=rrf_k,
            dense_weight=dense_weight,
            lexical_weight=lexical_weight,
        )[:limit]


def _format_result(result: SearchResult) -> dict[str, Any]:
    return {
        "chunk_id": result.chunk_id,
        "bid_id": result.metadata.get("bid_id"),
        "chunk_type": result.metadata.get("chunk_type"),
        "bid_title": result.metadata.get("bid_title"),
        "score": round(result.score, 6),
        "dense_rank": result.dense_rank,
        "lexical_rank": result.lexical_rank,
        "text": result.text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid dense + BM25 retrieval over GeM bid chunks.")
    parser.add_argument("--query", help="Natural-language query to retrieve.")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--chroma-path", default=str(DEFAULT_CHROMA_PATH))
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--lexical-db", help="Path for the FTS5 sidecar database.")
    parser.add_argument("--build-lexical-index", action="store_true")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild lexical index even when chunk count is unchanged.")
    parser.add_argument("--where", help='Chroma-style JSON metadata filter, e.g. {"ministry":"Defence"}.')
    parser.add_argument("--include-boilerplate", action="store_true")
    args = parser.parse_args()

    where = json.loads(args.where) if args.where else None
    retriever = HybridRetriever(args.chroma_path, args.collection, args.lexical_db)
    if args.build_lexical_index:
        indexed = retriever.build_lexical_index(rebuild=args.rebuild)
        print("Lexical index is current." if indexed == 0 else f"Built lexical index for {indexed:,} chunks.")
    if args.query:
        results = retriever.search(
            args.query,
            limit=args.limit,
            where=where,
            exclude_boilerplate=not args.include_boilerplate,
        )
        print(json.dumps([_format_result(result) for result in results], indent=2, ensure_ascii=False))
    elif not args.build_lexical_index:
        parser.error("provide --query and/or --build-lexical-index")


if __name__ == "__main__":
    main()