"""
Base vectorielle locale persistée — ChromaDB.
Stocke les documents extraits avec leurs embeddings et métadonnées.
Persiste dans data/vectorstore/ — survit aux redémarrages Streamlit.
"""
from __future__ import annotations
import hashlib
import json
from datetime import datetime
from pathlib import Path

STORE_DIR = Path(__file__).parent.parent.parent / "data" / "vectorstore"


def _get_collection():
    import chromadb
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(STORE_DIR))
    return client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"},
    )


def _doc_id(filename: str, page: int = 1) -> str:
    return hashlib.md5(f"{filename}:{page}".encode()).hexdigest()


def save_document(
    filename: str,
    doc_type: str,
    fields: dict,
    line_items: list,
    raw_text: str = "",
    page: int = 1,
    confidence: float = 0.0,
) -> str:
    """
    Persist an extracted document to ChromaDB.
    Returns the document ID.
    """
    col = _get_collection()
    doc_id = _doc_id(filename, page)

    # Text to embed = concatenation of key fields
    text_parts = [f"type:{doc_type}"]
    for k, v in fields.items():
        if v:
            text_parts.append(f"{k}:{v}")
    if raw_text:
        text_parts.append(raw_text[:500])
    embed_text = " | ".join(text_parts)

    metadata = {
        "filename":   filename,
        "page":       page,
        "doc_type":   doc_type,
        "confidence": confidence,
        "timestamp":  datetime.utcnow().isoformat(),
        "n_fields":   len(fields),
        "n_items":    len(line_items),
        "fields_json":      json.dumps(fields, ensure_ascii=False)[:1000],
        "line_items_json":  json.dumps(line_items, ensure_ascii=False)[:500],
    }

    # ChromaDB stores the text and uses its own default embedding function
    try:
        col.upsert(
            ids=[doc_id],
            documents=[embed_text],
            metadatas=[metadata],
        )
    except Exception as e:
        raise RuntimeError(f"ChromaDB save failed: {e}") from e

    return doc_id


def search_similar(query: str, n_results: int = 5) -> list[dict]:
    """Find documents similar to a text query."""
    col = _get_collection()
    if col.count() == 0:
        return []
    results = col.query(
        query_texts=[query],
        n_results=min(n_results, col.count()),
    )
    out = []
    for i, doc_id in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i]
        out.append({
            "id":         doc_id,
            "filename":   meta.get("filename"),
            "doc_type":   meta.get("doc_type"),
            "confidence": meta.get("confidence"),
            "timestamp":  meta.get("timestamp"),
            "n_fields":   meta.get("n_fields"),
            "fields":     json.loads(meta.get("fields_json", "{}")),
            "distance":   results["distances"][0][i] if results.get("distances") else None,
        })
    return out


def get_all_documents() -> list[dict]:
    """Return all persisted documents sorted by timestamp."""
    col = _get_collection()
    if col.count() == 0:
        return []
    results = col.get(include=["metadatas"])
    docs = []
    for i, doc_id in enumerate(results["ids"]):
        meta = results["metadatas"][i]
        docs.append({
            "id":        doc_id,
            "filename":  meta.get("filename"),
            "doc_type":  meta.get("doc_type"),
            "timestamp": meta.get("timestamp", ""),
            "n_fields":  meta.get("n_fields", 0),
            "fields":    json.loads(meta.get("fields_json", "{}")),
        })
    return sorted(docs, key=lambda x: x["timestamp"], reverse=True)


def count() -> int:
    return _get_collection().count()


def delete_document(doc_id: str):
    _get_collection().delete(ids=[doc_id])


def clear_all():
    import chromadb
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(STORE_DIR))
    client.delete_collection("documents")
