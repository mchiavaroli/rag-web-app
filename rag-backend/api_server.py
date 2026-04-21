import os
import sys
import glob
import json
import threading
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from fastapi.responses import JSONResponse  # ← aggiungi questa riga

# Imposta CWD nella directory del backend per risolvere i path relativi di config.py
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from config import (
    get_index_path, get_chunks_path, get_images_folder, get_metadata_path,
    OUTPUT_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, BATCH_SIZE,
    MIN_IMAGE_SIZE, ensure_output_dir
)

# ============================================================
# DIRECTORY SETUP  (percorsi assoluti per robustezza)
# ============================================================

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(_BASE_DIR, "docs")
IMAGES_DIR = os.path.join(_BASE_DIR, OUTPUT_DIR, "extracted_images")

os.makedirs(DOCS_DIR, exist_ok=True)
ensure_output_dir()
os.makedirs(IMAGES_DIR, exist_ok=True)

# ============================================================
# STATO GLOBALE
# ============================================================

indexing_state: dict = {
    "status": "idle",        # idle | building | ready | error
    "message": "Nessun documento caricato",
    "started_at": None,
    "completed_at": None,
    "total_chunks": 0,
    "text_chunks": 0,
    "image_chunks": 0,
}

_index_cache: dict = {
    "index": None,
    "chunks": None,
    "model": None,
}
_index_lock = threading.Lock()

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(title="RAG Backend API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve immagini estratte dai PDF (path assoluto)
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

# Serve i PDF originali (per il visualizzatore nel browser)
app.mount("/docs", StaticFiles(directory=DOCS_DIR), name="docs")

# ============================================================
# INDICIZZAZIONE IN BACKGROUND
# ============================================================

def _run_indexing() -> None:
    """Esegue la build dell'indice in un thread separato."""
    global indexing_state
    try:
        indexing_state["status"] = "building"
        indexing_state["started_at"] = datetime.now().isoformat()
        indexing_state["message"] = "Indicizzazione in corso..."

        # Invalida la cache
        with _index_lock:
            _index_cache["index"] = None
            _index_cache["chunks"] = None
            _index_cache["model"] = None

        from build_index import load_documents, build_index_multimodal_contextual

        docs = load_documents(DOCS_DIR)
        if not docs:
            indexing_state["status"] = "idle"
            indexing_state["message"] = "Nessun documento da indicizzare"
            return

        build_index_multimodal_contextual(
            docs,
            embed_model_name=EMBEDDING_MODEL,
            index_path=get_index_path(),
            meta_path=get_metadata_path(),
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
            extract_images=True,
            analyze_images=True,
            use_text_contextualization=True,
            batch_size=BATCH_SIZE,
            min_image_size=MIN_IMAGE_SIZE,
        )

        # Leggi le statistiche dall'indice appena creato
        chunks_path = get_chunks_path()
        if os.path.exists(chunks_path):
            all_chunks: dict = {}
            with open(chunks_path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    all_chunks[obj["chunk_id"]] = obj
            indexing_state["total_chunks"] = len(all_chunks)
            indexing_state["text_chunks"] = sum(1 for c in all_chunks.values() if c["type"] == "text")
            indexing_state["image_chunks"] = sum(1 for c in all_chunks.values() if c["type"] == "image")

        indexing_state["status"] = "ready"
        indexing_state["completed_at"] = datetime.now().isoformat()
        indexing_state["message"] = "Indice pronto"
        print("✅ Indicizzazione completata.")

    except Exception as exc:
        indexing_state["status"] = "error"
        indexing_state["message"] = f"Errore: {exc}"
        print(f"❌ Errore indicizzazione: {exc}")
        import traceback
        traceback.print_exc()


def _start_indexing() -> None:
    """Avvia l'indicizzazione in background (se non già in corso)."""
    if indexing_state["status"] != "building":
        threading.Thread(target=_run_indexing, daemon=True).start()


def _list_docs() -> List[dict]:
    """Restituisce la lista dei PDF nella cartella docs/."""
    docs = []
    for fp in glob.glob(os.path.join(DOCS_DIR, "*.pdf")):
        stat = os.stat(fp)
        name = os.path.basename(fp)
        docs.append({
            "id": name,
            "name": name,
            "size": stat.st_size,
            "uploadedAt": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "path": f"/docs/{name}",
            "indexed": indexing_state["status"] == "ready",
        })
    return sorted(docs, key=lambda x: x["uploadedAt"], reverse=True)


# ============================================================
# LOAD / CACHE INDICE
# ============================================================

def _get_or_load_index():
    """Carica l'indice FAISS dalla cache o dal disco."""
    with _index_lock:
        if _index_cache["index"] is not None:
            return _index_cache["index"], _index_cache["chunks"], _index_cache["model"]

        index_path = get_index_path()
        chunks_path = get_chunks_path()
        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            return None, None, None

        from rag_query import load_index
        index, chunks, model = load_index(index_path, chunks_path, EMBEDDING_MODEL)
        _index_cache["index"] = index
        _index_cache["chunks"] = chunks
        _index_cache["model"] = model
        return index, chunks, model


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup() -> None:
    """Controlla lo stato iniziale dell'indice all'avvio del server."""
    index_path = get_index_path()
    chunks_path = get_chunks_path()
    docs_count = len(glob.glob(os.path.join(DOCS_DIR, "*.pdf")))
    
    if os.path.exists(index_path) and os.path.exists(chunks_path):
        try:
            all_chunks: dict = {}
            indexed_docs: set = set()
            with open(chunks_path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    all_chunks[obj["chunk_id"]] = obj
                    indexed_docs.add(obj.get("source", ""))
            
            # Verifica: il numero di doc nell'indice corrisponde ai PDF fisici?
            if len(indexed_docs) < docs_count:
                print(f"⚠️  Documenti mismatch: {len(indexed_docs)} indicizzati, {docs_count} fisici.")
                print("📦 Avvio re-indicizzazione...")
                _start_indexing()
            else:
                indexing_state["status"] = "ready"
                indexing_state["message"] = "Indice pronto"
                indexing_state["total_chunks"] = len(all_chunks)
                indexing_state["text_chunks"] = sum(1 for c in all_chunks.values() if c["type"] == "text")
                indexing_state["image_chunks"] = sum(1 for c in all_chunks.values() if c["type"] == "image")
                print(f"✅ Indice trovato: {len(all_chunks)} chunk ({len(indexed_docs)} doc)")
        except Exception as exc:
            print(f"[WARN] Impossibile leggere lo stato dell'indice: {exc}")
            if docs_count > 0:
                _start_indexing()
    else:
        if docs_count > 0:
            print(f"📦 Trovati {docs_count} documenti — avvio indicizzazione automatica...")
            _start_indexing()
        else:
            indexing_state["status"] = "idle"
            indexing_state["message"] = "Nessun documento caricato"


# ============================================================
# MODELLI PYDANTIC
# ============================================================


# Support model selection
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    model: Optional[str] = None


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/api/status")
async def get_status() -> dict:
    """Restituisce lo stato dell'indice (status, chunk counts, documenti)."""
    return {**indexing_state, "documents": _list_docs()}


@app.get("/api/documents")
async def list_documents() -> dict:
    """Elenca i PDF presenti nella cartella docs/."""
    return {"documents": _list_docs()}


@app.get("/api/models")
def get_models():
    from config import MODEL_PROVIDERS, DEFAULT_MODEL_NAME
    return {
        "models": [
            {"id": name, **{k: v for k, v in cfg.items() if k != 'api_key'}}
            for name, cfg in MODEL_PROVIDERS.items()
        ],
        "default": DEFAULT_MODEL_NAME
    }


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)) -> dict:
    """Riceve un PDF, lo salva in docs/ e avvia l'indicizzazione."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo file PDF sono accettati.")

    # Sanitizza il nome per prevenire path traversal
    safe_name = Path(file.filename).name
    if not safe_name:
        raise HTTPException(status_code=400, detail="Nome file non valido.")

    dest_path = os.path.join(DOCS_DIR, safe_name)
    content = await file.read()
    with open(dest_path, "wb") as f:
        f.write(content)

    stat = os.stat(dest_path)
    doc = {
        "id": safe_name,
        "name": safe_name,
        "size": stat.st_size,
        "uploadedAt": datetime.now().isoformat(),
        "path": f"/docs/{safe_name}",
        "indexed": False,
    }
    _start_indexing()
    return {"success": True, "document": doc}


@app.delete("/api/documents/{filename}")
async def delete_document(filename: str) -> dict:
    """Elimina un PDF dalla cartella docs/ e aggiorna l'indice."""
    safe_name = Path(filename).name
    filepath = os.path.join(DOCS_DIR, safe_name)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Documento non trovato.")

    os.remove(filepath)

    # Invalida la cache
    with _index_lock:
        _index_cache["index"] = None
        _index_cache["chunks"] = None
        _index_cache["model"] = None

    remaining = glob.glob(os.path.join(DOCS_DIR, "*.pdf"))
    if remaining:
        _start_indexing()
    else:
        indexing_state["status"] = "idle"
        indexing_state["message"] = "Nessun documento"
        indexing_state["total_chunks"] = 0
        indexing_state["text_chunks"] = 0
        indexing_state["image_chunks"] = 0

    return {"success": True}


@app.post("/api/index")
async def trigger_index() -> dict:
    """Forza la re-indicizzazione di tutti i documenti."""
    if indexing_state["status"] == "building":
        return {"success": False, "message": "Indicizzazione già in corso."}

    docs = glob.glob(os.path.join(DOCS_DIR, "*.pdf"))
    if not docs:
        return {"success": False, "message": "Nessun documento da indicizzare."}

    _start_indexing()
    return {"success": True, "message": "Indicizzazione avviata."}


@app.post("/api/query")
async def query(request: QueryRequest) -> dict:
    """Esegue una query RAG e restituisce risposta + fonti."""
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query vuota.")

    cmd = request.query.strip().lower()

    # ── Comandi speciali (non richiedono indice) ─────────────────
    if cmd in ("stats", "statistiche"):
        docs = _list_docs()
        st = indexing_state
        lines = [
            "## 📊 Statistiche RAG",
            "",
            f"| Parametro | Valore |",
            f"|-----------|--------|",
            f"| Stato indice | **{st['status']}** |",
            f"| Documenti caricati | **{len(docs)}** |",
            f"| Chunk totali | **{st['total_chunks']}** |",
            f"| Chunk testo | {st['text_chunks']} |",
            f"| Chunk immagini | {st['image_chunks']} |",
        ]
        if st.get("started_at"):
            lines.append(f"| Ultima indicizzazione | {st['started_at'][:19].replace('T', ' ')} |")
        if st.get("completed_at"):
            lines.append(f"| Completata | {st['completed_at'][:19].replace('T', ' ')} |")
        if docs:
            lines += ["", "### Documenti", ""]
            for d in docs:
                lines.append(f"- **{d['name']}** ({d['size'] // 1024} KB)")
        return {"answer": "\n".join(lines), "sources": []}

    if cmd in ("logs", "log"):
        log_path = os.path.join(_BASE_DIR, "output", "logs", "query_logs.jsonl")
        entries = []
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        recent = entries[-10:]
        if not recent:
            return {"answer": "## 📋 Logs\n\nNessun log disponibile.", "sources": []}
        lines = ["## 📋 Log query recenti", ""]
        for e in reversed(recent):
            ts = e.get("timestamp", "")[:19].replace("T", " ")
            q_text = e.get("query", {}).get("text", "—")[:60]
            success = "✅" if e.get("success") else "❌"
            latency = e.get("total_latency_ms", 0)
            chunks = e.get("retrieval", {}).get("chunks_retrieved", 0)
            lines.append(f"**{ts}** {success} — *{q_text}*")
            lines.append(f"&nbsp;&nbsp;&nbsp;&nbsp;Latenza: {latency} ms · Chunk recuperati: {chunks}")
            lines.append("")
        return {"answer": "\n".join(lines), "sources": []}

    if cmd in ("history", "cronologia"):
        log_path = os.path.join(_BASE_DIR, "output", "logs", "query_logs.jsonl")
        entries = []
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        recent = entries[-15:]
        if not recent:
            return {"answer": "## 🕐 Cronologia\n\nNessuna query registrata.", "sources": []}
        lines = ["## 🕐 Ultime query", ""]
        for i, e in enumerate(reversed(recent), 1):
            ts = e.get("timestamp", "")[:19].replace("T", " ")
            q_text = e.get("query", {}).get("text", "—")
            lines.append(f"{i}. **{ts}** — {q_text}")
        return {"answer": "\n".join(lines), "sources": []}

    # ── Query RAG normale ────────────────────────────────────────
    if indexing_state["status"] == "building":
        raise HTTPException(
            status_code=503,
            detail="Indicizzazione in corso. Riprova tra qualche minuto.",
        )

    if indexing_state["status"] != "ready":
        raise HTTPException(
            status_code=503,
            detail="Nessun indice disponibile. Carica dei documenti e attendi l'indicizzazione.",
        )

    index, chunks, model = _get_or_load_index()
    if index is None:
        raise HTTPException(status_code=503, detail="Impossibile caricare l'indice.")


    from rag_query import ask_llm
    from config import get_model_provider, MODEL_PROVIDERS, DEFAULT_MODEL_NAME

    # Scegli il modello richiesto, fallback a default
    model_name = request.model or DEFAULT_MODEL_NAME
    model_config = get_model_provider(model_name)
    if not model_config:
        model_config = MODEL_PROVIDERS[DEFAULT_MODEL_NAME]

    response_text, retrieved, images_shown = ask_llm(
        query=request.query,
        index=index,
        chunks=chunks,
        model=model,
        show_sources=False,
        enable_logging=True,
        session_id=request.session_id,
        model_config=model_config,
    )

    # ── Fonti: testo da retrieved + immagini da images_shown (loggato) ──
    sources = []
    seen: set = set()
    
    # Aggiungi fonti testo
    for result in retrieved:
        chunk = result["chunk"]
        if chunk["type"] == "text":
            key = f"text_{chunk['source']}_{chunk.get('chunk_id', '')}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "type": "pdf",
                    "path": f"/docs/{chunk['source']}",
                    "title": chunk["source"],
                    "page": chunk.get("page"),
                    "preview": chunk.get("text_original", "")[:250],
                })
    
    # Aggiungi SOLO le immagini che sono state loggatte (images_shown)
    # Cerchiamo il chunk corrispondente per ottenere source e page
    chunk_by_image_name = {}
    for result in retrieved:
        chunk = result["chunk"]
        if chunk["type"] == "image":
            img_filename = os.path.basename(chunk.get("image_path", ""))
            chunk_by_image_name[img_filename] = chunk
    
    for img_info in images_shown:
        img_filename = img_info["name"]
        key = f"image_{img_filename}"
        if key not in seen:
            seen.add(key)
            chunk = chunk_by_image_name.get(img_filename)
            if chunk:
                abs_img_path = os.path.join(IMAGES_DIR, img_filename)
                if os.path.exists(abs_img_path):
                    sources.append({
                        "type": "image",
                        "path": f"/images/{img_filename}",
                        "title": chunk.get("source"),
                        "page": chunk.get("page"),
                        "preview": chunk.get("text_original", "")[:200],
                    })

    return {"answer": response_text, "sources": sources}


@app.delete("/api/logs")
async def delete_logs() -> dict:
    """Elimina il file di log delle query."""
    log_path = os.path.join(_BASE_DIR, "output", "logs", "query_logs.jsonl")
    try:
        if os.path.exists(log_path):
            os.remove(log_path)
            return {"success": True, "message": "Log eliminati"}
        else:
            return {"success": True, "message": "Nessun log da eliminare"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore eliminazione log: {e}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  RAG Backend API Server")
    print("  http://localhost:8000")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
