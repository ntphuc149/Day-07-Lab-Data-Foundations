from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        sentences = re.split(r'(?<=\.)\s+|(?<=[!?])\s+|(?<=\.)\n', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[i : i + self.max_sentences_per_chunk]
            chunks.append(" ".join(group))
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if len(current_text) <= self.chunk_size:
            return [current_text]

        # No separators left — force-split by character
        if not remaining_separators:
            return [current_text[i : i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        sep = remaining_separators[0]
        rest = remaining_separators[1:]

        if sep == "":
            # Character-level fallback
            return [current_text[i : i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        parts = current_text.split(sep)
        result: list[str] = []
        current_piece = ""

        for part in parts:
            candidate = (current_piece + sep + part) if current_piece else part
            if len(candidate) <= self.chunk_size:
                current_piece = candidate
            else:
                if current_piece:
                    result.extend(self._split(current_piece, rest))
                current_piece = part

        if current_piece:
            result.extend(self._split(current_piece, rest))

        return result if result else [current_text]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    mag_a = math.sqrt(_dot(vec_a, vec_a))
    mag_b = math.sqrt(_dot(vec_b, vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return _dot(vec_a, vec_b) / (mag_a * mag_b)


class LegalArticleChunker:
    """
    DFS chunker cho văn bản pháp luật Việt Nam theo cấu trúc Chương → Điều.

    Cấp 1 — tách theo Chương: bắt "**Chương X**" + tên chương ở dòng tiếp theo.
    Cấp 2 — trong mỗi Chương, tách theo Điều: "**Điều N. Tên điều**".
    Mỗi chunk được prefix "Chương X — Tên | Điều Y. Tên" để giữ đủ context.
    Fallback — nếu 1 Điều vẫn > chunk_size, dùng RecursiveChunker.
    """

    # Bắt: **Chương X** + toàn bộ dòng tên chương (có thể nhiều dòng **...**)
    # Dừng khi gặp **Điều hoặc hết block **...**
    _CHUONG = re.compile(
        r'(\*\*Chương\s+\S+\*\*(?:\n\n\*\*(?!Điều)[^*]+\*\*)+)',
        re.UNICODE,
    )
    # Bắt: **Điều N. Tên điều**
    _DIEU = re.compile(r'(\*\*Điều\s+\d+\.[^*]+\*\*)', re.UNICODE)

    def __init__(self, chunk_size: int = 500) -> None:
        self.chunk_size = chunk_size
        self._fallback = RecursiveChunker(chunk_size=chunk_size)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        parts = self._CHUONG.split(text)
        result: list[str] = []
        current_chuong = ""

        for part in parts:
            if self._CHUONG.fullmatch(part.strip()):
                current_chuong = part.strip()
            else:
                result.extend(self._split_chuong(current_chuong, part))

        return [c for c in result if c.strip()]

    def _split_chuong(self, chuong_header: str, body: str) -> list[str]:
        # findall trả về list [(dieu_header, content), ...]
        # Dùng re.split với capturing group: phần tử lẻ là header, chẵn là content
        tokens = self._DIEU.split(body)
        result: list[str] = []
        current_dieu = ""

        i = 0
        while i < len(tokens):
            token = tokens[i]
            if self._DIEU.fullmatch(token):
                current_dieu = token
            else:
                content = token.strip()
                if content:
                    prefix_parts = []
                    if chuong_header:
                        prefix_parts.append(chuong_header)
                    if current_dieu:
                        prefix_parts.append(current_dieu)
                    prefix = "\n".join(prefix_parts)
                    chunk = (prefix + "\n\n" + content) if prefix else content
                    result.extend(self._split_if_needed(chunk))
            i += 1

        return result

    def _split_if_needed(self, chunk: str) -> list[str]:
        if len(chunk) <= self.chunk_size:
            return [chunk]
        # Giữ 2 dòng header (Chương + Điều) làm prefix cho mỗi fallback chunk
        lines = chunk.splitlines()
        header_lines = [l for l in lines[:4] if l.startswith("**")]
        prefix = "\n".join(header_lines)
        body_start = chunk.find("\n\n", len(prefix))
        body = chunk[body_start:].strip() if body_start != -1 else chunk
        sub_chunks = self._fallback.chunk(body)
        return [(prefix + "\n\n" + s) if prefix else s for s in sub_chunks if s.strip()]


class LegalChunker:
    """
    Custom chunker for legal texts (Vietnamese) — by Trần Minh Anh.

    Strategy:
      - First split by 'Điều <number>' headings.
      - If a 'Điều' section is still too long, split by 'Khoản' or numbered subclauses.
      - Fallback: fixed-size slicing.
    """

    def __init__(self, chunk_size: int = 1200) -> None:
        self.chunk_size = max(100, int(chunk_size))

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        text = text.strip()
        parts = re.split(r'(?m)(?:(?<=\n\n)|^)(?=Điều\s*\d+\b)', text)
        if len(parts) <= 1:
            parts = re.split(r'(?m)(?:(?<=\n\n)|^)(?=\d+\.)', text)
        chunks: list[str] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) <= self.chunk_size:
                chunks.append(part)
                continue
            subparts = re.split(r'(?m)(?=^\s*(?:Khoản\b|\d+\.)\s*)', part)
            if len(subparts) > 1:
                for sub in subparts:
                    sub = sub.strip()
                    if not sub:
                        continue
                    if len(sub) <= self.chunk_size:
                        chunks.append(sub)
                    else:
                        for i in range(0, len(sub), self.chunk_size):
                            piece = sub[i : i + self.chunk_size].strip()
                            if piece:
                                chunks.append(piece)
            else:
                for i in range(0, len(part), self.chunk_size):
                    piece = part[i : i + self.chunk_size].strip()
                    if piece:
                        chunks.append(piece)
        return chunks


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size).chunk(text),
            "by_sentences": SentenceChunker().chunk(text),
            "recursive": RecursiveChunker(chunk_size=chunk_size).chunk(text),
        }
        result = {}
        for name, chunks in strategies.items():
            count = len(chunks)
            avg_length = sum(len(c) for c in chunks) / count if count else 0.0
            result[name] = {"count": count, "avg_length": avg_length, "chunks": chunks}
        return result
