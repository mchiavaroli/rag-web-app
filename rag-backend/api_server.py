import os
import sys
import glob
import json
import re
import threading
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
    get_chunks_path, get_images_folder,
    OUTPUT_DIR, CHUNK_SIZE, CHUNK_OVERLAP, BATCH_SIZE,
    MIN_IMAGE_SIZE, ensure_output_dir, MODEL_PROMPT,
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
    """Estrae testo e immagini dai documenti e salva i chunk in JSONL per la wiki."""
    global indexing_state

    try:
        indexing_state["status"] = "building"
        indexing_state["started_at"] = datetime.now().isoformat()
        indexing_state["message"] = "Indicizzazione in corso..."

        from build_index_document_intelligence import (
            load_documents_with_document_intelligence,
            build_index_with_document_intelligence,
        )

        docs = load_documents_with_document_intelligence(DOCS_DIR, use_doc_intelligence=True)
        if not docs:
            indexing_state["status"] = "idle"
            indexing_state["message"] = "Nessun documento da indicizzare"
            return

        build_index_with_document_intelligence(
            docs,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
            analyze_images=True,
            use_text_contextualization=True,
            batch_size=BATCH_SIZE,
            min_image_size=MIN_IMAGE_SIZE,
            use_document_intelligence=True,
        )

        # Aggiorna contatori dallo JSONL appena creato
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
        indexing_state["message"] = "Indicizzazione completata"
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
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup() -> None:
    """Controlla lo stato iniziale all'avvio del server."""
    chunks_path = get_chunks_path()
    docs_count = len(glob.glob(os.path.join(DOCS_DIR, "*.pdf")))

    if os.path.exists(chunks_path):
        try:
            all_chunks: dict = {}
            indexed_docs: set = set()
            with open(chunks_path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    all_chunks[obj["chunk_id"]] = obj
                    indexed_docs.add(obj.get("source", ""))

            if len(indexed_docs) < docs_count:
                print(f"⚠️  Documenti mismatch: {len(indexed_docs)} indicizzati, {docs_count} fisici.")
                print("📦 Avvio re-indicizzazione...")
                _start_indexing()
            else:
                indexing_state["status"] = "ready"
                indexing_state["message"] = "Indicizzazione pronta"
                indexing_state["total_chunks"] = len(all_chunks)
                indexing_state["text_chunks"] = sum(1 for c in all_chunks.values() if c["type"] == "text")
                indexing_state["image_chunks"] = sum(1 for c in all_chunks.values() if c["type"] == "image")
                print(f"✅ Chunks trovati: {len(all_chunks)} ({len(indexed_docs)} doc)")
        except Exception as exc:
            print(f"[WARN] Impossibile leggere i chunks: {exc}")
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



# ============================================================
# WIKI ENDPOINTS
# ============================================================

class WikiQueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    model: Optional[str] = None


class WikiIngestRequest(BaseModel):
    filename: Optional[str] = None  # Se omesso, ingerisce tutti i documenti


class WikiLearnRequest(BaseModel):
    text: str
    hint_title: Optional[str] = None
    model: Optional[str] = None


@app.get("/api/wiki/status")
async def wiki_status() -> dict:
    """Restituisce lo stato della wiki."""
    from wiki_manager import get_wiki_status
    return get_wiki_status()


@app.get("/api/wiki/pages")
async def wiki_pages() -> dict:
    """Elenca tutte le pagine wiki."""
    from wiki_manager import list_wiki_pages
    return {"pages": list_wiki_pages()}


@app.get("/api/wiki/graph")
async def wiki_graph() -> dict:
    """Restituisce nodi e archi della wiki per visualizzazione a grafo."""
    from wiki_manager import get_wiki_graph
    return get_wiki_graph()


@app.get("/api/wiki/pages/{category}/{filename}")
async def wiki_page(category: str, filename: str) -> dict:
    """Legge una singola pagina wiki."""
    from wiki_manager import get_wiki_page
    page = get_wiki_page(category, filename)
    if not page:
        raise HTTPException(status_code=404, detail="Pagina wiki non trovata.")
    return page


@app.post("/api/wiki/ingest")
async def wiki_ingest(request: WikiIngestRequest = WikiIngestRequest()) -> dict:
    """Ingerisce documenti nella wiki. Se filename è specificato, ingerisce solo quello."""
    from wiki_manager import ingest_document
    from config import get_model_provider, DEFAULT_MODEL_NAME

    docs_to_ingest = []

    if request.filename:
        # Ingerisci un singolo documento
        safe_name = Path(request.filename).name
        filepath = os.path.join(DOCS_DIR, safe_name)
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="Documento non trovato.")
        docs_to_ingest.append(safe_name)
    else:
        # Ingerisci tutti i documenti
        docs_to_ingest = [os.path.basename(f) for f in glob.glob(os.path.join(DOCS_DIR, "*.pdf"))]

    if not docs_to_ingest:
        return {"success": False, "message": "Nessun documento da ingerire."}

    # Legge i chunk già elaborati dal file JSONL dell'indice (più veloce, evita di richiamare DI)
    from config import get_chunks_path

    results = []
    for doc_name in docs_to_ingest:
        try:
            chunks_path = get_chunks_path()
            if not os.path.exists(chunks_path):
                results.append({"filename": doc_name, "success": False, "error": "Indice non trovato. Esegui prima l'indicizzazione."})
                continue

            # Filtra i chunk del documento richiesto dal JSONL
            doc_chunks = []
            with open(chunks_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if obj.get("source") == doc_name:
                            doc_chunks.append(obj)
                    except json.JSONDecodeError:
                        continue

            if not doc_chunks:
                results.append({"filename": doc_name, "success": False, "error": f"Nessun chunk trovato per '{doc_name}'. Verifica che il documento sia stato indicizzato."})
                continue

            # Ricostruisci testo e immagini dai chunk già estratti
            text_parts = []
            images_info = []
            for chunk in doc_chunks:
                if chunk["type"] in ("text", "table"):
                    # Usa text_contextualized se disponibile (più ricco), altrimenti text_original
                    text = chunk.get("text_contextualized") or chunk.get("text_original", "")
                    if text:
                        text_parts.append(text)
                elif chunk["type"] == "image":
                    images_info.append({
                        "path": chunk.get("image_path", ""),
                        "description": chunk.get("text_original", ""),
                        "page": chunk.get("page"),
                    })

            doc_text = "\n\n".join(text_parts)

            if not doc_text.strip():
                results.append({"filename": doc_name, "success": False, "error": "Testo estratto vuoto."})
                continue

            result = ingest_document(
                doc_text=doc_text,
                doc_name=doc_name,
                images_info=images_info,
            )
            results.append({"filename": doc_name, **result})

        except Exception as exc:
            results.append({"filename": doc_name, "success": False, "error": str(exc)})

    total_created = sum(r.get("pages_created", 0) for r in results)
    total_updated = sum(r.get("pages_updated", 0) for r in results)

    return {
        "success": True,
        "documents_processed": len(results),
        "total_pages_created": total_created,
        "total_pages_updated": total_updated,
        "results": results,
    }


@app.post("/api/wiki/learn")
async def wiki_learn_endpoint(request: WikiLearnRequest) -> dict:
    """Impara un concetto da testo libero e aggiorna la wiki."""
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Testo vuoto.")

    from wiki_manager import ingest_text
    from config import get_model_provider, DEFAULT_MODEL_NAME

    model_name = request.model or DEFAULT_MODEL_NAME
    model_config = get_model_provider(model_name)

    result = ingest_text(
        text=request.text,
        hint_title=request.hint_title,
        model_config=model_config,
    )
    return result


@app.post("/api/wiki/query")
async def wiki_query_endpoint(request: WikiQueryRequest) -> dict:
    """Esegue una query usando la wiki (non i documenti raw)."""
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query vuota.")

    from wiki_manager import wiki_query as wq
    from config import get_model_provider, DEFAULT_MODEL_NAME

    model_name = request.model or DEFAULT_MODEL_NAME
    model_config = get_model_provider(model_name) or MODEL_PROMPT

    answer, pages_used, usage, wiki_context = wq(
        query=request.query,
        session_history=None,
        model_config=model_config,
    )

    # --- Fonti wiki (pagine markdown) ---
    sources = []
    seen: set = set()
    for page_path in pages_used:
        key = f"wiki_{page_path}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "type": "wiki",
                "path": page_path,
                "title": page_path.split("/")[-1].replace(".md", "").replace("-", " ").title(),
            })

    # --- Fonti PDF: estrai riferimenti a .pdf citati nella risposta ---
    pdf_refs = re.findall(r'([\w\s()\-\[\].]+\.pdf)', answer, re.IGNORECASE)
    for pdf_name in dict.fromkeys(pdf_refs):
        pdf_name = pdf_name.strip()
        key = f"pdf_{pdf_name}"
        if key not in seen and os.path.exists(os.path.join(DOCS_DIR, pdf_name)):
            seen.add(key)
            sources.append({
                "type": "pdf",
                "path": f"/docs/{pdf_name}",
                "title": pdf_name,
            })

    return {"answer": answer, "sources": sources}


@app.post("/api/wiki/lint")
async def wiki_lint() -> dict:
    """Esegue un audit/lint della wiki."""
    from wiki_manager import lint_wiki
    return lint_wiki()


@app.post("/api/wiki/fix")
async def wiki_fix(body: dict) -> dict:
    """Applica le correzioni suggerite dall'audit Lint."""
    from wiki_manager import fix_wiki
    issues = body.get("issues", [])
    suggestions = body.get("suggestions", [])
    return fix_wiki(issues=issues, suggestions=suggestions)


@app.get("/api/wiki/log")
async def wiki_log_endpoint() -> dict:
    """Restituisce il log delle operazioni wiki."""
    from wiki_manager import get_wiki_log
    return {"log": get_wiki_log()}


@app.delete("/api/wiki")
async def wiki_reset() -> dict:
    """Resetta la wiki (elimina tutte le pagine generate)."""
    from wiki_manager import reset_wiki
    return reset_wiki()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  RAG Backend API Server")
    print("  http://localhost:8000")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
