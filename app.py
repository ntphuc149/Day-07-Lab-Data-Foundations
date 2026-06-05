from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template_string, request

load_dotenv(Path(".env"), override=False)

from src.chunking import (
    FixedSizeChunker,
    LegalArticleChunker,
    RecursiveChunker,
    SentenceChunker,
)
from src.agent import KnowledgeBaseAgent
from src.embeddings import OpenAIEmbedder, _mock_embed
from src.models import Document
from src.store import EmbeddingStore

app = Flask(__name__)

# ── Global state ──────────────────────────────────────────────────────────────
_store: EmbeddingStore | None = None
_agent: KnowledgeBaseAgent | None = None
_loaded_files: list[str] = []
_current_strategy: str = "legal_article"

TEMPLATE = Path(__file__).parent / "demo.html"


def _get_chunker(strategy: str, chunk_size: int = 500):
    if strategy == "fixed_size":
        return FixedSizeChunker(chunk_size=chunk_size)
    if strategy == "sentence":
        return SentenceChunker(max_sentences_per_chunk=3)
    if strategy == "recursive":
        return RecursiveChunker(chunk_size=chunk_size)
    return LegalArticleChunker(chunk_size=chunk_size)


def _get_embedder():
    key = os.environ.get("OPENAI_API_KEY", "")
    if key and key != "your-key-here":
        try:
            return OpenAIEmbedder()
        except Exception:
            pass
    return _mock_embed


def _rebuild_store(files_content: dict[str, str], strategy: str, chunk_size: int = 500):
    global _store, _agent
    embedder = _get_embedder()
    chunker = _get_chunker(strategy, chunk_size)
    store = EmbeddingStore(collection_name="demo", embedding_fn=embedder)
    docs = []
    for filename, content in files_content.items():
        doc_id = Path(filename).stem
        chunks = chunker.chunk(content)
        for i, chunk in enumerate(chunks):
            docs.append(Document(
                id=f"{doc_id}__{i}",
                content=chunk,
                metadata={"doc_id": doc_id, "filename": filename, "chunk_index": i},
            ))
    store.add_documents(docs)

    def llm_fn(prompt: str) -> str:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key or api_key == "your-key-here":
            chunks_in_ctx = [l for l in prompt.split("\n") if l.startswith("[")]
            return "⚠️ Chưa có OpenAI API key — context retrieved:\n" + "\n".join(chunks_in_ctx[:3])
        # Parse system / user blocks từ prompt format của agent
        import re as _re
        sys_match = _re.search(r"<system>\n(.*?)\n</system>", prompt, _re.DOTALL)
        usr_match = _re.search(r"<user>\n(.*?)\n</user>", prompt, _re.DOTALL)
        system_content = sys_match.group(1) if sys_match else ""
        user_content = usr_match.group(1) if usr_match else prompt
        from openai import OpenAI
        client = OpenAI()
        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": user_content})
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=512,
        )
        return resp.choices[0].message.content

    _store = store
    _agent = KnowledgeBaseAgent(store=store, llm_fn=llm_fn)
    return len(docs)


# ── In-memory file cache ──────────────────────────────────────────────────────
_file_cache: dict[str, str] = {}


@app.route("/")
def index():
    return TEMPLATE.read_text(encoding="utf-8")


@app.route("/api/upload", methods=["POST"])
def upload():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files"}), 400
    for f in files:
        content = f.read().decode("utf-8")
        _file_cache[f.filename] = content
    strategy = request.form.get("strategy", _current_strategy)
    chunk_size = int(request.form.get("chunk_size", 500))
    n = _rebuild_store(_file_cache, strategy, chunk_size)
    return jsonify({"files": list(_file_cache.keys()), "total_chunks": n})


@app.route("/api/rebuild", methods=["POST"])
def rebuild():
    if not _file_cache:
        return jsonify({"error": "No files loaded"}), 400
    data = request.json or {}
    strategy = data.get("strategy", "legal_article")
    chunk_size = int(data.get("chunk_size", 500))
    n = _rebuild_store(_file_cache, strategy, chunk_size)
    return jsonify({"total_chunks": n, "strategy": strategy})


@app.route("/api/preview", methods=["POST"])
def preview():
    data = request.json or {}
    filename = data.get("filename")
    strategy = data.get("strategy", "legal_article")
    chunk_size = int(data.get("chunk_size", 500))
    if not filename or filename not in _file_cache:
        return jsonify({"error": "File not found"}), 404
    chunker = _get_chunker(strategy, chunk_size)
    chunks = chunker.chunk(_file_cache[filename])
    return jsonify({
        "filename": filename,
        "strategy": strategy,
        "count": len(chunks),
        "avg_length": round(sum(len(c) for c in chunks) / len(chunks), 1) if chunks else 0,
        "chunks": [{"index": i, "length": len(c), "text": c} for i, c in enumerate(chunks)],
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    if not _agent:
        return jsonify({"error": "Chưa upload file nào"}), 400
    data = request.json or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400
    top_k = int(data.get("top_k", 3))
    results = _store.search(question, top_k=top_k)
    answer = _agent.answer(question, top_k=top_k)
    sources = [
        {
            "doc_id": r["metadata"].get("doc_id"),
            "chunk_index": r["metadata"].get("chunk_index"),
            "score": round(r["score"], 4),
            "preview": r["content"][:200],
        }
        for r in results
    ]
    return jsonify({"answer": answer, "sources": sources})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
