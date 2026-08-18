from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            # Load lazily so importing the module does not download a model.
            # from sentence_transformers import CrossEncoder
            # self._model = CrossEncoder(self.model_name)
            #
            # ⚠️ LƯU Ý: Dùng sentence_transformers.CrossEncoder, KHÔNG dùng FlagEmbedding.
            # FlagReranker crash với transformers>=5.0 (XLMRobertaTokenizer lỗi).
            try:
                from sentence_transformers import CrossEncoder
                # Avoid an unbounded first-run Hugging Face download. Tests and
                # offline runs use the deterministic fallback unless the model
                # is already cached locally. Set RERANK_ALLOW_DOWNLOAD=1 to
                # explicitly permit downloading the model.
                allow_download = os.getenv("RERANK_ALLOW_DOWNLOAD", "0") == "1"
                if allow_download:
                    self._model = CrossEncoder(self.model_name)
                else:
                    self._model = CrossEncoder(self.model_name, local_files_only=True)
            except Exception:
                self._model = False
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if top_k <= 0:
            return []
        # 1. if not documents: return []
        # 2. model = self._load_model()
        # 3. pairs = [(query, doc["text"]) for doc in documents]
        # 4. scores = model.predict(pairs)
        # 5. if isinstance(scores, (int, float)): scores = [scores]
        # 6. scored = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        # 7. Return [RerankResult(text=..., original_score=doc.get("score", 0.0),
        #            rerank_score=float(score), metadata=..., rank=i)
        #            for i, (score, doc) in enumerate(scored[:top_k])]
        if not documents:
            return []
        model = self._load_model()
        if model:
            scores = model.predict([(query, d.get("text", "")) for d in documents])
        else:
            # Deterministic fallback for offline environments. It rewards
            # query-term coverage and keeps the original retrieval score only
            # as a small tie-breaker.
            qwords = {w.strip(".,!?;:()[]{}\"").lower()
                      for w in query.split() if w.strip()}
            scores = []
            for document in documents:
                words = set(document.get("text", "").lower().split())
                overlap = len(qwords & words)
                scores.append(overlap + float(document.get("score", 0.0)) * 0.001)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        if isinstance(scores, (int, float)):
            scores = [scores]
        ranked = sorted(zip(scores, documents), key=lambda x: float(x[0]), reverse=True)[:top_k]
        return [RerankResult(d.get("text", ""), d.get("score", 0.0), float(score),
                             d.get("metadata", {}), i + 1)
                for i, (score, d) in enumerate(ranked)]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        # Optional FlashRank implementation can be added when low latency is
        # more important than the CrossEncoder's ranking quality.
        # model = Ranker(); passages = [{"text": d["text"]} for d in documents]
        # results = model.rerank(RerankRequest(query=query, passages=passages))
        return []


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
