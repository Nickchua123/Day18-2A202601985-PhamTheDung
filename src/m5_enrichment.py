from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys, json, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, OPENAI_MODEL, get_openai_client


def _parse_json_object(value: str) -> dict | None:
    """Parse JSON even when a provider wraps it in Markdown or extra prose."""
    if not value:
        return None
    cleaned = value.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned,
                     flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(cleaned[start:end + 1])
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def _local_enrichment(text: str, source: str) -> dict:
    """Deterministic fallback that makes no additional API calls."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    summary = " ".join(sentences[:2]) or text[:300]
    questions = [f"{sentence.rstrip('.!?')}?" for sentence in sentences[:3]]
    return {
        "summary": summary,
        "questions": questions,
        "context": f"Trích từ {source}." if source else "",
        "metadata": {"topic": "general", "entities": [],
                     "category": "policy", "language": "vi"},
    }


def _chat(prompt: str, system: str, max_tokens: int = 200) -> str:
    """Call the configured OpenAI-compatible provider."""
    client = get_openai_client()
    if client is None:
        return ""
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    # Prefer the LLM, then use an extractive local fallback.
    # if OPENAI_API_KEY:
    #     try:
    #         from openai import OpenAI
    #         client = OpenAI()
    #         resp = client.chat.completions.create(
    #             model="gpt-4o-mini",
    #             messages=[
    #                 {"role": "system", "content": "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt."},
    #                 {"role": "user", "content": text},
    #             ],
    #             max_tokens=150,
    #         )
    #         return resp.choices[0].message.content.strip()
    #     except Exception as e:
    #         print(f"  ⚠️  OpenAI summarize failed: {e}")
    #
    # Extractive fallback (không cần API):
    # sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    # return ". ".join(sentences[:2]) + "." if sentences else text
    try:
        result = _chat(text, "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt.", 150)
        if result:
            return result
    except Exception as e:
        print(f"  ⚠️  LLM summarize failed: {e}")
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    return ". ".join(sentences[:2]) + ("." if sentences else "")


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    # Prefer the LLM, then derive questions from the first sentences.
    # if OPENAI_API_KEY:
    #     try:
    #         from openai import OpenAI
    #         client = OpenAI()
    #         resp = client.chat.completions.create(
    #             model="gpt-4o-mini",
    #             messages=[
    #                 {"role": "system", "content": f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. Trả về mỗi câu hỏi trên 1 dòng."},
    #                 {"role": "user", "content": text},
    #             ],
    #             max_tokens=200,
    #         )
    #         questions = resp.choices[0].message.content.strip().split("\n")
    #         return [q.strip().lstrip("0123456789.-) ") for q in questions if q.strip()][:n_questions]
    #     except Exception as e:
    #         print(f"  ⚠️  OpenAI HyQA failed: {e}")
    #
    # Extractive fallback:
    # import re
    # sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if len(s.strip()) > 10]
    # return [f"{s.rstrip('.')}?" for s in sentences[:n_questions]]
    try:
        result = _chat(text, f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. Mỗi câu hỏi trên một dòng.", 200)
        if result:
            return [q.strip().lstrip("0123456789.-) ") for q in result.splitlines() if q.strip()][:n_questions]
    except Exception as e:
        print(f"  ⚠️  LLM HyQA failed: {e}")
    import re
    sentences = [s.strip() for s in re.split(r"[.!?\n]", text) if len(s.strip()) > 10]
    return [f"{s.rstrip('.')}?" for s in sentences[:n_questions]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    # Prefer the LLM, then retain the source title as context.
    # if OPENAI_API_KEY:
    #     try:
    #         from openai import OpenAI
    #         client = OpenAI()
    #         resp = client.chat.completions.create(
    #             model="gpt-4o-mini",
    #             messages=[
    #                 {"role": "system", "content": "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. Chỉ trả về 1 câu."},
    #                 {"role": "user", "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"},
    #             ],
    #             max_tokens=80,
    #         )
    #         context = resp.choices[0].message.content.strip()
    #         return f"{context}\n\n{text}"
    #     except Exception as e:
    #         print(f"  ⚠️  OpenAI contextual failed: {e}")
    #
    # Simple fallback:
    # prefix = f"Trích từ {document_title}. " if document_title else ""
    # return f"{prefix}{text}"
    try:
        context = _chat(
            f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}",
            "Viết một câu ngắn mô tả đoạn văn nằm ở đâu trong tài liệu và nói về chủ đề gì. Chỉ trả về một câu.",
            80,
        )
        if context:
            return f"{context}\n\n{text}"
    except Exception as e:
        print(f"  ⚠️  LLM contextual failed: {e}")
    prefix = f"Trích từ {document_title}. " if document_title else ""
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    # Prefer the LLM, then return stable default metadata.
    # if OPENAI_API_KEY:
    #     try:
    #         import json as _json
    #         from openai import OpenAI
    #         client = OpenAI()
    #         resp = client.chat.completions.create(
    #             model="gpt-4o-mini",
    #             messages=[
    #                 {"role": "system", "content": 'Trích xuất metadata từ đoạn văn. Trả về JSON: {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}'},
    #                 {"role": "user", "content": text},
    #             ],
    #             max_tokens=150,
    #         )
    #         return _json.loads(resp.choices[0].message.content)
    #     except Exception as e:
    #         print(f"  ⚠️  OpenAI metadata failed: {e}")
    #
    # return {"topic": "general", "entities": [], "category": "policy", "language": "vi"}
    try:
        result = _chat(
            text,
            'Trích xuất metadata và chỉ trả về JSON: {"topic":"...", "entities":[], "category":"policy|hr|it|finance", "language":"vi|en"}',
            150,
        )
        if result:
            parsed = _parse_json_object(result)
            if parsed:
                return parsed
            print("  ⚠️  LLM metadata returned invalid JSON; using fallback.")
    except Exception as e:
        print(f"  ⚠️  LLM metadata failed: {e}")
    return {"topic": "general", "entities": [], "category": "policy", "language": "vi"}


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    # One call per chunk in the normal path; fallback is local-only.
    # if OPENAI_API_KEY:
    #     try:
    #         import json as _json
    #         from openai import OpenAI
    #         client = OpenAI()
    #         resp = client.chat.completions.create(
    #             model="gpt-4o-mini",
    #             messages=[
    #                 {"role": "system", "content": """Phân tích đoạn văn và trả về JSON:
    # {
    #   "summary": "tóm tắt 2-3 câu",
    #   "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
    #   "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu",
    #   "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
    # }"""},
    #                 {"role": "user", "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}"},
    #             ],
    #             max_tokens=400,
    #         )
    #         return _json.loads(resp.choices[0].message.content)
    #     except Exception as e:
    #         print(f"  ⚠️  Enrichment API failed: {e}")
    try:
        result = _chat(
            f"Tài liệu: {source}\n\nĐoạn văn:\n{text}",
            """Phân tích đoạn văn và chỉ trả về JSON hợp lệ với cấu trúc:
{"summary":"...","questions":["..."],"context":"...","metadata":{"topic":"...","entities":[],"category":"policy|hr|it|finance","language":"vi|en"}}""",
            400,
        )
        if result:
            parsed = _parse_json_object(result)
            if parsed:
                return parsed
            print("  ⚠️  LLM enrichment returned invalid JSON; using fallback.")
    except Exception as e:
        print(f"  ⚠️  LLM enrichment failed: {e}")
    return _local_enrichment(text, source)


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
