from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    # Use underthesea when available; keep a plain-text fallback for minimal
    # environments where the optional tokenizer is unavailable.
    # 1. from underthesea import word_tokenize
    # 2. segmented = word_tokenize(text, format="text")
    # 3. return segmented.replace("_", " ")
    #
    # ⚠️ LƯU Ý: underthesea nối từ ghép bằng "_" (VD: "nghỉ_phép").
    # BM25 tokenize bằng split(" ") → "nghỉ_phép" thành 1 token,
    # nhưng query "nghỉ phép" thành 2 token → KHÔNG khớp.
    # Phải replace("_", " ") để BM25 hoạt động đúng.
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text").replace("_", " ")
    except Exception:
        return text


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        # Store the original chunks so search results can return full text and
        # metadata, while BM25 works on segmented tokens.
        # 1. self.documents = chunks
        # 2. For each chunk: segment_vietnamese(chunk["text"]) → split by space
        # 3. self.corpus_tokens = [tokenized list for each chunk]
        # 4. from rank_bm25 import BM25Okapi
        #    self.bm25 = BM25Okapi(self.corpus_tokens)
        from rank_bm25 import BM25Okapi
        self.documents = chunks
        self.corpus_tokens = [segment_vietnamese(c["text"]).lower().split() for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        # Return only positive matches: zero-score documents add noise to the
        # hybrid candidate set and make an empty/no-match query misleading.
        # 1. if self.bm25 is None: return []
        # 2. tokenized_query = segment_vietnamese(query).split()
        # 3. scores = self.bm25.get_scores(tokenized_query)
        # 4. top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        # 5. Return [SearchResult(text=..., score=..., metadata=..., method="bm25")]
        #    Lọc scores[i] > 0 để bỏ docs không liên quan.
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(segment_vietnamese(query).lower().split())
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [SearchResult(self.documents[i]["text"], float(scores[i]), self.documents[i].get("metadata", {}), "bm25")
                for i in indices[:top_k] if scores[i] > 0]


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, PointStruct, VectorParams

        if not chunks:
            self.client.recreate_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=EMBEDDING_DIM,
                                            distance=Distance.COSINE),
            )
            return

        texts = [str(chunk.get("text", "")) for chunk in chunks]
        vectors = self._get_encoder().encode(
            texts, show_progress_bar=False, normalize_embeddings=True
        )
        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM,
                                        distance=Distance.COSINE),
        )
        points = []
        for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
            payload = {**chunk.get("metadata", {}), "text": texts[index]}
            points.append(PointStruct(id=index, vector=vector.tolist(), payload=payload))
        self.client.upsert(collection_name=collection, points=points)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if top_k <= 0:
            return []
        try:
            query_vector = self._get_encoder().encode(
                query, normalize_embeddings=True
            ).tolist()
            response = self.client.query_points(
                collection_name=collection, query=query_vector, limit=top_k
            )
        except Exception:
            # A missing collection or unavailable local Qdrant should not make
            # the caller crash; BM25 can still provide useful results.
            return []
        results = []
        for point in getattr(response, "points", []):
            payload = point.payload or {}
            text = payload.get("text", "")
            metadata = {key: value for key, value in payload.items() if key != "text"}
            results.append(SearchResult(text=text, score=float(point.score),
                                        metadata=metadata, method="dense"))
        return results


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    if top_k <= 0:
        return []
    k = max(0, k)
    # 1. rrf_scores = {}  # text → {"score": float, "result": SearchResult}
    # 2. For each result_list in results_list:
    #      For rank, result in enumerate(result_list):
    #        if result.text not in rrf_scores: rrf_scores[result.text] = {"score": 0.0, "result": result}
    #        rrf_scores[result.text]["score"] += 1.0 / (k + rank + 1)
    # 3. Sort by score descending
    # 4. Return top_k SearchResult with method="hybrid"
    scores, seen = {}, {}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            key = result.text
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            seen.setdefault(key, result)
    ordered = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [SearchResult(key, scores[key], seen[key].metadata, "hybrid") for key in ordered]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
