"""
Wiki Manager — Gestisce la knowledge base wiki (Layer 2 dell'LLM Wiki pattern).

Tre operazioni principali:
1. INGEST: Documento → LLM legge + genera/aggiorna pagine wiki
2. QUERY: Domanda → LLM legge wiki pre-sintetizzata → risposta
3. LINT:  Audit periodico della wiki (contraddizioni, orfani, gap)
"""
import os
import re
import json
import glob
import time
from datetime import datetime
from typing import Optional
import base64
import hashlib
import mimetypes
from config import (
    get_wiki_dir, get_wiki_schema_path, get_wiki_index_path, get_wiki_log_path,
    WIKI_MAX_CONTEXT_PAGES, WIKI_INGEST_MAX_TOKENS, WIKI_INGEST_MAX_BATCHES,
    MODEL_PROMPT, get_model_provider, DEFAULT_MODEL_NAME,
)
from llm_client import call_llm_text, call_llm_with_image


# ============================================================
# UTILITY
# ============================================================

WIKI_CATEGORIES = ["sources", "concepts", "procedures", "components", "images"]


def _ensure_wiki_dirs():
    """Crea le directory wiki se non esistono."""
    wiki_dir = get_wiki_dir()
    for sub in WIKI_CATEGORIES:
        os.makedirs(os.path.join(wiki_dir, sub), exist_ok=True)


def _read_file(path: str) -> str:
    """Legge un file e restituisce il contenuto, o stringa vuota se non esiste."""
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write_file(path: str, content: str):
    """Scrive un file, creando le directory necessarie."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _sanitize_filename(name: str) -> str:
    """Converte un nome in kebab-case safe per filesystem."""
    name = name.lower().strip()
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'[\s_]+', '-', name)
    name = re.sub(r'-+', '-', name)
    return name.strip('-')


# ============================================================
# WIKI STATE
# ============================================================

def get_wiki_status() -> dict:
    """Restituisce lo stato della wiki: conteggio pagine per categoria."""
    wiki_dir = get_wiki_dir()
    _ensure_wiki_dirs()

    categories = {}
    for sub in WIKI_CATEGORIES:
        sub_dir = os.path.join(wiki_dir, sub)
        pages = glob.glob(os.path.join(sub_dir, "*.md"))
        categories[sub] = len(pages)

    total = sum(categories.values())
    has_index = os.path.exists(get_wiki_index_path())

    return {
        "status": "ready" if total > 0 else "empty",
        "total_pages": total,
        "categories": categories,
        "has_index": has_index,
    }


def list_wiki_pages() -> list:
    """Elenca tutte le pagine wiki con metadati."""
    wiki_dir = get_wiki_dir()
    _ensure_wiki_dirs()
    pages = []

    for sub in WIKI_CATEGORIES:
        sub_dir = os.path.join(wiki_dir, sub)
        for fp in glob.glob(os.path.join(sub_dir, "*.md")):
            stat = os.stat(fp)
            name = os.path.basename(fp)
            content = _read_file(fp)

            # Estrai titolo (prima riga che inizia con #)
            title = name.replace(".md", "").replace("-", " ").title()
            for line in content.split("\n"):
                if line.startswith("# "):
                    title = line[2:].strip()
                    break

            # Conta wikilinks
            links = re.findall(r'\[\[([^\]]+)\]\]', content)

            pages.append({
                "name": name,
                "category": sub,
                "title": title,
                "path": f"wiki/{sub}/{name}",
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "links_count": len(links),
                "links": links,
            })

    return sorted(pages, key=lambda x: x["modified"], reverse=True)


def get_wiki_graph() -> dict:
    """Restituisce nodi e archi per la visualizzazione a grafo della wiki."""
    pages = list_wiki_pages()

    nodes = []
    edges = []
    seen_edges: set = set()
    node_ids: set = set()

    for page in pages:
        slug = page['name'].replace('.md', '')
        node_ids.add(slug)
        nodes.append({
            'id': slug,
            'label': page['title'],
            'category': page['category'],
            'links_count': page.get('links_count', 0),
        })

    def _add_edge(src: str, dst: str):
        if dst not in node_ids:
            return
        key = f"{src}→{dst}"
        if key not in seen_edges:
            seen_edges.add(key)
            edges.append({'id': key, 'source': src, 'target': dst})

    for page in pages:
        source_slug = page['name'].replace('.md', '')
        wiki_dir = get_wiki_dir()
        safe_cat = os.path.basename(page['category'])
        safe_name = os.path.basename(page['name'])
        content = _read_file(os.path.join(wiki_dir, safe_cat, safe_name))

        # Archi da [[wikilink]]
        for target_slug in re.findall(r'\[\[([^\]]+)\]\]', content):
            _add_edge(source_slug, target_slug)

        # Archi da riferimenti immagine: `images/<nomefile>`
        # I file .md delle immagini usano tutti dash; i riferimenti nei testi
        # possono usare underscore (es. _p10_img1). Normalizziamo _ → - per il lookup.
        for img_ref in re.findall(r'`images/([^`]+)`', content):
            img_slug = re.sub(r'\.[^.]+$', '', img_ref)   # rimuovi estensione
            img_slug = img_slug.replace('_', '-')           # normalizza _ → -
            _add_edge(source_slug, img_slug)

    return {'nodes': nodes, 'edges': edges}


def get_wiki_page(category: str, filename: str) -> Optional[dict]:
    """Legge una singola pagina wiki."""
    wiki_dir = get_wiki_dir()
    # Sanitize: previeni path traversal
    safe_cat = os.path.basename(category)
    safe_name = os.path.basename(filename)
    filepath = os.path.join(wiki_dir, safe_cat, safe_name)

    if not os.path.exists(filepath):
        return None

    content = _read_file(filepath)
    title = safe_name.replace(".md", "").replace("-", " ").title()
    for line in content.split("\n"):
        if line.startswith("# "):
            title = line[2:].strip()
            break

    links = re.findall(r'\[\[([^\]]+)\]\]', content)

    return {
        "name": safe_name,
        "category": safe_cat,
        "title": title,
        "content": content,
        "links": links,
    }


# ============================================================
# WIKI CONTEXT BUILDER
# ============================================================

def _load_all_wiki_content(max_pages: int = None) -> str:
    """Carica tutte le pagine wiki come contesto per il LLM."""
    if max_pages is None:
        max_pages = WIKI_MAX_CONTEXT_PAGES

    wiki_dir = get_wiki_dir()
    _ensure_wiki_dirs()
    pages_content = []

    # Carica index e schema per primi
    index_content = _read_file(get_wiki_index_path())
    if index_content:
        pages_content.append(f"=== FILE: wiki/index.md ===\n{index_content}")

    count = 0
    for sub in WIKI_CATEGORIES:
        sub_dir = os.path.join(wiki_dir, sub)
        for fp in sorted(glob.glob(os.path.join(sub_dir, "*.md"))):
            if count >= max_pages:
                break
            content = _read_file(fp)
            name = os.path.basename(fp)
            pages_content.append(f"=== FILE: wiki/{sub}/{name} ===\n{content}")
            count += 1

    return "\n\n" + "\n\n".join(pages_content)


def _find_relevant_pages(query: str, max_pages: int = 8) -> str:
    """Trova le pagine wiki più rilevanti per una query (match per keyword)."""
    wiki_dir = get_wiki_dir()
    _ensure_wiki_dirs()

    query_words = set(w.lower() for w in re.findall(r'\w{3,}', query))
    scored_pages = []

    for sub in WIKI_CATEGORIES:
        sub_dir = os.path.join(wiki_dir, sub)
        for fp in glob.glob(os.path.join(sub_dir, "*.md")):
            content = _read_file(fp)
            content_lower = content.lower()
            name = os.path.basename(fp)

            # Score: conta quante keyword della query appaiono nel contenuto
            score = sum(1 for w in query_words if w in content_lower)
            # Bonus per match nel titolo/nome file
            name_lower = name.lower().replace("-", " ").replace(".md", "")
            score += sum(2 for w in query_words if w in name_lower)

            if score > 0:
                scored_pages.append((score, sub, name, content))

    scored_pages.sort(key=lambda x: x[0], reverse=True)
    top_pages = scored_pages[:max_pages]

    if not top_pages:
        # Fallback: carica tutte le pagine
        return _load_all_wiki_content(max_pages)

    parts = []
    # Sempre includere l'index
    index_content = _read_file(get_wiki_index_path())
    if index_content:
        parts.append(f"=== FILE: wiki/index.md ===\n{index_content}")

    for _, sub, name, content in top_pages:
        parts.append(f"=== FILE: wiki/{sub}/{name} ===\n{content}")

    return "\n\n" + "\n\n".join(parts)


# ============================================================
# INGEST — Compila un documento nella wiki
# ============================================================

def ingest_document(doc_text: str, doc_name: str, images_info: list = None,
                    model_config: dict = None) -> dict:
    """
    Ingerisce un documento nella wiki: il LLM legge il contenuto e genera/aggiorna
    le pagine wiki appropriate.

    Per documenti grandi il testo viene suddiviso in chunk e ogni chunk viene
    processato in una chiamata LLM separata, accumulando le pagine prodotte.

    Args:
        doc_text: testo estratto dal documento (via Document Intelligence)
        doc_name: nome del file sorgente (es. "manual_v2.pdf")
        images_info: lista di dict con info immagini [{path, description, page}]
        model_config: config del modello LLM da usare (default: MODEL_PROMPT)

    Returns:
        dict con info sull'operazione {pages_created, pages_updated, log_entry}
    """
    _ensure_wiki_dirs()
    base_config = model_config or MODEL_PROMPT
    config = dict(base_config)
    # Forza almeno 8192 token di output per evitare troncamenti del JSON
    if config.get('provider') == 'openai':
        config['max_completion_tokens'] = max(config.get('max_completion_tokens', 0), 8192)
    else:
        config['max_tokens'] = max(config.get('max_tokens', 0), 8192)

    schema = _read_file(get_wiki_schema_path())
    wiki_dir = get_wiki_dir()

    # ── Split del testo in chunk per documenti grandi ──────────────────────
    CHUNK_CHARS = WIKI_INGEST_MAX_TOKENS * 4   # ~32 768 caratteri ≈ 8 k token
    OVERLAP_CHARS = 500                         # sovrapposizione tra chunk adiacenti

    if len(doc_text) <= CHUNK_CHARS:
        text_chunks = [doc_text]
    else:
        text_chunks = []
        start = 0
        while start < len(doc_text):
            text_chunks.append(doc_text[start:start + CHUNK_CHARS])
            if start + CHUNK_CHARS >= len(doc_text):
                break
            start += CHUNK_CHARS - OVERLAP_CHARS

    raw_chunks = len(text_chunks)

    # ── Cap: se i batch superano il limite, campiona per copertura uniforme ──
    if len(text_chunks) > WIKI_INGEST_MAX_BATCHES:
        step = (len(text_chunks) - 1) / (WIKI_INGEST_MAX_BATCHES - 1)
        text_chunks = [text_chunks[round(i * step)] for i in range(WIKI_INGEST_MAX_BATCHES)]
        print(
            f"📚 [wiki_ingest] '{doc_name}' — documento molto grande: "
            f"{raw_chunks} segmenti → campionati {WIKI_INGEST_MAX_BATCHES} "
            f"(copertura uniforme, {len(doc_text):,} caratteri totali)."
        )
    else:
        print(f"📚 [wiki_ingest] '{doc_name}' → {len(text_chunks)} chunk(s) da processare.")

    n_chunks = len(text_chunks)

    # ── Mappa pagina → slug immagini (calcolata deterministicamente) ────────
    # Usa la stessa logica di ingest_images_to_wiki per generare i slug,
    # così il LLM può inserire [[wikilink]] corretti anche prima che le pagine esistano.
    page_image_map: dict = {}  # {page_num_str: [(slug, img_fname), ...]}
    if images_info:
        for img in images_info:
            img_fname = os.path.basename(img.get('path', ''))
            if not img_fname:
                continue
            slug = _sanitize_filename(os.path.splitext(img_fname)[0])
            page_key = str(img.get('page', '?'))
            page_image_map.setdefault(page_key, []).append((slug, img_fname))

    # Sezione immagini compatta con slug navigabili — inclusa in TUTTI i chunk
    images_map_section = ""
    if page_image_map:
        map_lines = []
        for page_key in sorted(page_image_map.keys(), key=lambda x: int(x) if x.isdigit() else 0):
            slugs_info = page_image_map[page_key]
            links = ", ".join(f"[[{s}]]" for s, _ in slugs_info)
            map_lines.append(f"- Pagina {page_key}: {links}")
        images_map_section = (
            "\n\n## Mappa Immagini del Documento (usa questi [[link]] in 'Riferimenti Visivi')\n"
            + "\n".join(map_lines)
            + "\n\nREGOLA: Ogni pagina di componente/concetto/procedura che fa riferimento a figure "
              "deve includere una sezione '## Riferimenti Visivi' con i [[wikilink]] "
              "corrispondenti alla pagina del documento. Es: se un componente è descritto a pag. 103, "
              "aggiungi `## Riferimenti Visivi\n- [[t-roc-2024-...p103-img1]]`"
        )

    system_prompt = f"""Sei un wiki maintainer esperto. Segui rigorosamente queste regole:

{schema}

IMPORTANTE:
- Rispondi SOLO con un JSON valido. Nessun testo prima o dopo il JSON.
- Ogni campo "content" deve essere CONCISO: massimo 150-200 parole per pagina.
- Non ripetere informazioni già presenti in altre pagine, usa [[link]] invece.
- Il campo "index_update" deve contenere solo l'indice aggiornato, non copiare l'intero contenuto delle pagine."""

    pages_created = 0
    pages_updated = 0
    total_llm_time = 0.0
    total_usage: dict = {}
    last_log_entry = f"Ingerito: {doc_name}"
    last_index_update = None

    # ── Analisi vision immagini PRIMA del loop testuale ────────────────────
    # Così le pagine wiki/images/ esistono già quando il LLM scrive i componenti
    # e le trova nella existing_wiki → può aggiungere [[link]] corretti.
    images_result = {"pages_created": 0, "pages_updated": 0, "skipped": 0}
    if images_info:
        from config import get_images_folder
        print(f"🖼️  [wiki_ingest] Analisi vision di {len(images_info)} immagini prima dell'ingest testuale...")
        images_result = ingest_images_to_wiki(
            images_info=images_info,
            doc_name=doc_name,
            images_base_dir=get_images_folder(),
            model_config=model_config,
        )
        pages_created += images_result['pages_created']
        pages_updated += images_result['pages_updated']
        print(f"  → {images_result['pages_created']} pagine immagine create, {images_result['skipped']} skip")

    # ── Loop sui chunk ─────────────────────────────────────────────────────
    for chunk_idx, chunk_text in enumerate(text_chunks):
        is_first = chunk_idx == 0
        is_last = chunk_idx == n_chunks - 1

        # Ricarica la wiki ad ogni iterazione: il chunk successivo vede le pagine già scritte
        existing_wiki = _load_all_wiki_content()

        chunk_note = ""
        if n_chunks > 1:
            chunk_note = (
                f"\n\n**NOTA — Segmento {chunk_idx + 1}/{n_chunks}:** "
                + ("Crea la pagina sommario in `sources/` solo in questo primo segmento. " if is_first
                   else "NON ricreare la pagina sommario (già esistente). ")
                + ("Aggiorna `index_update` con tutti i link (incluse pagine già nella wiki)." if is_last
                   else "Per `index_update` usa **null**: l'indice verrà aggiornato solo nell'ultimo segmento.")
            )

        user_prompt = f"""## Operazione: INGEST{f' — segmento {chunk_idx + 1}/{n_chunks}' if n_chunks > 1 else ''}

### Documento da ingerire
**Nome file:** {doc_name}

**Contenuto estratto:**
{chunk_text}
{images_map_section}
{chunk_note}

### Wiki esistente
{existing_wiki}

### Istruzioni
Analizza il testo e genera pagine wiki per i concetti/procedure/componenti presenti IN QUESTO SEGMENTO.
Rispondi con un JSON con questa struttura esatta:

```json
{{
  "pages": [
    {{
      "action": "create" | "update",
      "category": "sources" | "concepts" | "procedures" | "components",
      "filename": "nome-file.md",
      "content": "# Titolo\\n\\n> Una riga di descrizione.\\n\\n## Dettagli\\nContenuto conciso (max 150 parole)...\\n\\n## Riferimenti Visivi\\n- [[slug-immagine-pagina]]\\n\\n## Fonti\\n- Documento: {doc_name}"
    }}
  ],
  "index_update": {"null" if not is_last else '"# Wiki Index\\n\\n## Fonti\\n- [[nome-pagina]]\\n\\n## Concetti\\n- [[altro]]"'},
  "log_entry": "Breve descrizione dell'operazione"
}}
```

REGOLE CRITICHE:
- Ogni "content" MAX 150 parole. Usa elenchi puntati, non prosa.
- Crea pagine SEPARATE per ogni concetto/procedura/componente (non accumularli in una sola).
- Usa `[[link]]` per collegare invece di ripetere info.
- **IMMAGINI**: Ogni pagina che descrive qualcosa illustrato nel documento DEVE includere `## Riferimenti Visivi` con i [[wikilink]] corretti dalla "Mappa Immagini" sopra, basandoti sul numero di pagina dove appare il concetto.
- {"index_update: aggiorna l'indice con TUTTI i link (incluse pagine già presenti nella wiki)." if is_last else "index_update: deve essere null."}

Genera{":" if not is_first else " (solo per questo primo segmento):"}
{("1. Una pagina sommario in `sources/` per questo documento" + chr(10)) if is_first else ""}{"2" if is_first else "1"}. Pagine per ogni concetto tecnico in `concepts/` (una per concetto)
{"3" if is_first else "2"}. Pagine per procedure operative in `procedures/` (una per procedura)
{"4" if is_first else "3"}. Pagine per componenti/Part Numbers in `components/` (una per componente)
{("5. Aggiorna `index.md` con i nuovi link") if is_last else ""}"""

        t0 = time.time()
        response_text, usage = call_llm_text(config, system_prompt, user_prompt)
        chunk_llm_time = (time.time() - t0) * 1000
        total_llm_time += chunk_llm_time

        # Accumula usage tokens
        for k, v in (usage or {}).items():
            if isinstance(v, (int, float)):
                total_usage[k] = total_usage.get(k, 0) + v

        result = _parse_ingest_response(response_text, doc_name)

        # Scrivi le pagine immediatamente (il prossimo chunk le vedrà in existing_wiki)
        for page in result.get("pages", []):
            cat = page.get("category", "concepts")
            fname = page.get("filename", "untitled.md")
            if not fname.endswith(".md"):
                fname += ".md"

            filepath = os.path.join(wiki_dir, cat, fname)
            existed = os.path.exists(filepath)
            _write_file(filepath, page.get("content", ""))

            if existed:
                pages_updated += 1
            else:
                pages_created += 1

        if result.get("index_update"):
            last_index_update = result["index_update"]
        if result.get("log_entry"):
            last_log_entry = result["log_entry"]

        print(f"  chunk {chunk_idx + 1}/{n_chunks}: {len(result.get('pages', []))} pagine — {chunk_llm_time:.0f}ms")

    # Aggiorna index.md con l'ultimo index_update valido
    if last_index_update:
        _write_file(get_wiki_index_path(), last_index_update)

    # Aggiorna log.md
    log_entry = last_log_entry
    _append_log(f"**INGEST** — {log_entry}")

    return {
        "success": True,
        "pages_created": pages_created,
        "pages_updated": pages_updated,
        "total_pages": pages_created + pages_updated,
        "images_analyzed": len(images_info) if images_info else 0,
        "images_pages_created": images_result['pages_created'],
        "images_skipped": images_result['skipped'],
        "log_entry": log_entry,
        "chunks_processed": n_chunks,
        "llm_time_ms": round(total_llm_time, 2),
        "usage": total_usage,
    }


def _parse_ingest_response(response_text: str, doc_name: str) -> dict:
    """Estrae il JSON dalla risposta del LLM (gestisce markdown code blocks)."""
    # Rimuovi code block markdown se presente
    text = response_text.strip()
    if text.startswith("```"):
        # Rimuovi prima e ultima riga (```json e ```)
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]

    # Tentativo 1: parse diretto
    try:
        result = json.loads(text)
        if isinstance(result, dict) and "pages" in result:
            return result
    except json.JSONDecodeError:
        pass

    # Tentativo 2: cerca il JSON nell'output (il LLM a volte aggiunge testo prima/dopo)
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, dict) and "pages" in result:
                return result
        except json.JSONDecodeError:
            pass

    # Tentativo 3: il LLM potrebbe aver scritto solo l'array "pages" - cerca il blocco pages
    pages_match = re.search(r'"pages"\s*:\s*(\[[\s\S]*?\])', text)
    if pages_match:
        try:
            pages = json.loads(pages_match.group(1))
            return {"pages": pages, "index_update": None, "log_entry": f"Ingerito {doc_name} (parsing parziale)"}
        except json.JSONDecodeError:
            pass

    # Fallback: log del raw output per debug + crea pagina minimale
    print(f"⚠️  [wiki_manager] JSON parsing fallito per '{doc_name}'. Output LLM (primi 500 chars):\n{response_text[:500]}")
    safe_name = _sanitize_filename(doc_name.replace(".pdf", ""))
    return {
        "pages": [{
            "action": "create",
            "category": "sources",
            "filename": f"{safe_name}.md",
            "content": f"# {doc_name}\n\n> Sommario generato automaticamente.\n\n{response_text[:2000]}"
        }],
        "index_update": None,
        "log_entry": f"Ingerito {doc_name} (parsing fallito — solo pagina sorgente)",
    }


# ============================================================
# INGEST IMMAGINI — Analisi vision + pagine wiki per ogni immagine
# ============================================================

def ingest_images_to_wiki(images_info: list, doc_name: str, images_base_dir: str,
                          model_config: dict = None) -> dict:
    """
    Per ogni immagine del documento, chiama il LLM vision e crea una pagina wiki in images/.

    Args:
        images_info:    lista [{path, description, page}] dove path è il path assoluto o relativo
        doc_name:       nome del documento sorgente
        images_base_dir: directory base dove risiedono le immagini (per i path relativi)
        model_config:   config LLM da usare (deve supportare vision, es. MODEL_PROMPT claude)

    Returns:
        dict con {pages_created, pages_updated, skipped}
    """
    _ensure_wiki_dirs()
    wiki_dir = get_wiki_dir()
    config = dict(model_config or MODEL_PROMPT)
    # Usa max_tokens moderato per le descrizioni di singole immagini
    if config.get('provider') == 'openai':
        config['max_completion_tokens'] = min(config.get('max_completion_tokens', 1024), 1024)
    else:
        config['max_tokens'] = min(config.get('max_tokens', 1024), 1024)

    pages_created = 0
    pages_updated = 0
    skipped = 0
    seen_hashes: dict = {}  # hash -> img_filename già processato (dedup immagini identiche)

    for img in images_info:
        img_path = img.get('path', '')
        if not img_path:
            skipped += 1
            continue

        # Risolvi il path assoluto
        if os.path.isabs(img_path):
            abs_path = img_path
        else:
            # path relativo: prova prima da images_base_dir, poi da cwd
            abs_path = os.path.join(images_base_dir, os.path.basename(img_path))
            if not os.path.exists(abs_path):
                abs_path = img_path  # tentativo con path originale

        if not os.path.exists(abs_path):
            print(f"⚠️  [wiki_manager] Immagine non trovata, skip: {img_path}")
            skipped += 1
            continue

        # Leggi e codifica in base64
        try:
            with open(abs_path, 'rb') as f:
                img_bytes = f.read()
            # Deduplicazione per hash MD5: se l'immagine è identica a una già processata, skip
            img_hash = hashlib.md5(img_bytes).hexdigest()
            if img_hash in seen_hashes:
                print(f"⏭️  [wiki_manager] Skip immagine duplicata: {os.path.basename(img_path)} == {seen_hashes[img_hash]}")
                skipped += 1
                continue
            seen_hashes[img_hash] = os.path.basename(img_path)

            b64 = base64.b64encode(img_bytes).decode('utf-8')
            mime, _ = mimetypes.guess_type(abs_path)
            if not mime or not mime.startswith('image/'):
                mime = 'image/png'
        except Exception as e:
            print(f"⚠️  [wiki_manager] Errore lettura immagine {img_path}: {e}")
            skipped += 1
            continue

        img_filename = os.path.basename(img_path)
        page_num = img.get('page', '?')
        existing_desc = img.get('description', '')

        # Prompt vision: analisi tecnica dettagliata dell'immagine
        vision_prompt = (
            f"Sei un tecnico automotive. Analizza questa immagine tecnica dal documento '{doc_name}' (pagina {page_num}).\n"
            f"Descrivi in modo CONCISO (max 120 parole) cosa mostra: componenti visibili, operazioni rappresentate, "
            f"numeri di parte, frecce/etichette, sequenze operative. "
            f"Se è un diagramma di assemblaggio, descrivi i passi mostrati. "
            f"Usa un linguaggio tecnico preciso."
        )
        if existing_desc:
            vision_prompt += f"\n\nDescrizione OCR esistente (da usare come contesto): {existing_desc[:300]}"

        try:
            vision_desc = call_llm_with_image(config, b64, mime, vision_prompt)
        except Exception as e:
            print(f"⚠️  [wiki_manager] Errore vision LLM per {img_filename}: {e}")
            vision_desc = existing_desc or "Analisi visiva non disponibile."

        # Crea la pagina wiki per questa immagine
        # Il path `images/{filename}` viene embeddato nella pagina
        # così il regex in api_server.py lo estrae automaticamente come fonte
        safe_name = _sanitize_filename(os.path.splitext(img_filename)[0])
        wiki_fname = f"{safe_name}.md"
        wiki_path = os.path.join(wiki_dir, 'images', wiki_fname)
        existed = os.path.exists(wiki_path)

        page_content = (
            f"# Figura: {img_filename}\n\n"
            f"> Immagine tecnica da `{doc_name}` — pagina {page_num}.\n\n"
            f"## File\n`images/{img_filename}`\n\n"
            f"## Analisi Visiva\n{vision_desc}\n\n"
            f"## Documento Sorgente\n{doc_name} — pagina {page_num}\n"
        )
        _write_file(wiki_path, page_content)

        if existed:
            pages_updated += 1
        else:
            pages_created += 1

    _append_log(
        f"**INGEST IMMAGINI** — {doc_name}: "
        f"{pages_created} create, {pages_updated} aggiornate, {skipped} skip"
    )

    return {
        'pages_created': pages_created,
        'pages_updated': pages_updated,
        'skipped': skipped,
    }


# ============================================================
# LEARN — Impara da testo libero in linguaggio naturale
# ============================================================

def ingest_text(text: str, hint_title: str = None, model_config: dict = None) -> dict:
    """
    Impara un concetto da testo libero scritto dall'utente e genera/aggiorna
    le pagine wiki appropriate.

    Args:
        text: testo in linguaggio naturale scritto dall'utente
        hint_title: titolo suggerito dall'utente (opzionale)
        model_config: config del modello LLM da usare (default: MODEL_PROMPT)

    Returns:
        dict con info sull'operazione {pages_created, pages_updated, log_entry}
    """
    _ensure_wiki_dirs()

    base_config = model_config or MODEL_PROMPT
    config = dict(base_config)
    if config.get('provider') == 'openai':
        config['max_completion_tokens'] = max(config.get('max_completion_tokens', 0), 4096)
    else:
        config['max_tokens'] = max(config.get('max_tokens', 0), 4096)

    existing_wiki = _load_all_wiki_content()
    schema = _read_file(get_wiki_schema_path())

    title_hint = f"\n**Titolo suggerito dall'utente:** {hint_title}" if hint_title else ""

    system_prompt = f"""Sei un wiki maintainer esperto. Il tuo compito è integrare nuove conoscenze
fornite dall'utente in linguaggio naturale nella wiki esistente.

{schema}

IMPORTANTE:
- Rispondi SOLO con un JSON valido. Nessun testo prima o dopo il JSON.
- Ogni campo "content" deve essere CONCISO: massimo 150-200 parole per pagina.
- Non ripetere informazioni già presenti in altre pagine, usa [[link]] invece.
- Identifica automaticamente la categoria più adatta (concepts, procedures, components).
- Se il concetto esiste già, aggiorna la pagina esistente invece di crearne una nuova."""

    user_prompt = f"""## Operazione: LEARN (input utente)

### Nuovo concetto da integrare
{title_hint}

**Testo fornito dall'utente:**
{text.strip()}

### Wiki esistente
{existing_wiki}

### Istruzioni
Analizza il testo fornito dall'utente e integra le informazioni nella wiki.
Rispondi con un JSON con questa struttura esatta:

```json
{{
  "pages": [
    {{
      "action": "create" | "update",
      "category": "concepts" | "procedures" | "components",
      "filename": "nome-file.md",
      "content": "# Titolo\\n\\n> Breve descrizione.\\n\\n## Dettagli\\nContenuto conciso (max 150 parole)...\\n\\n## Note\\n- Aggiunto manualmente dall'utente"
    }}
  ],
  "index_update": "# Wiki Index\\n\\n## Concetti\\n- [[nome-pagina]]",
  "log_entry": "Breve descrizione del concetto aggiunto"
}}
```

REGOLE:
- Crea al massimo 2-3 pagine (non frammentare eccessivamente).
- Scegli la categoria giusta: concepts (cosa è), procedures (come si fa), components (parte/sistema).
- Se il concetto è già presente nella wiki, usa action "update" con il filename esistente.
- Aggiorna index_update solo se stai creando pagine nuove."""

    t0 = time.time()
    response_text, usage = call_llm_text(config, system_prompt, user_prompt)
    llm_time_ms = (time.time() - t0) * 1000

    result = _parse_ingest_response(response_text, hint_title or "user-input")

    wiki_dir = get_wiki_dir()
    pages_created = 0
    pages_updated = 0

    for page in result.get("pages", []):
        cat = page.get("category", "concepts")
        if cat not in ("concepts", "procedures", "components"):
            cat = "concepts"
        fname = page.get("filename", "untitled.md")
        if not fname.endswith(".md"):
            fname += ".md"

        filepath = os.path.join(wiki_dir, cat, fname)
        existed = os.path.exists(filepath)
        _write_file(filepath, page.get("content", ""))

        if existed:
            pages_updated += 1
        else:
            pages_created += 1

    if result.get("index_update"):
        _write_file(get_wiki_index_path(), result["index_update"])

    log_entry = result.get("log_entry", f"Appreso da input utente: {hint_title or text[:50]}")
    _append_log(f"**LEARN** — {log_entry}")

    return {
        "success": True,
        "pages_created": pages_created,
        "pages_updated": pages_updated,
        "total_pages": pages_created + pages_updated,
        "log_entry": log_entry,
        "llm_time_ms": round(llm_time_ms, 2),
        "usage": usage,
    }


# ============================================================
# QUERY — Rispondi usando la wiki
# ============================================================

def wiki_query(query: str, session_history: list = None,
               model_config: dict = None) -> tuple:
    """
    Risponde a una domanda usando la wiki pre-compilata (non i documenti raw).

    Args:
        query: domanda dell'utente
        session_history: cronologia conversazione [{role, content}]
        model_config: config del modello LLM

    Returns:
        (answer_text, wiki_pages_used, usage)
    """
    config = model_config or MODEL_PROMPT

    # Trova pagine rilevanti
    wiki_context = _find_relevant_pages(query)

    if not wiki_context.strip() or wiki_context.strip() == "":
        return ("La wiki è vuota. Carica dei documenti e usa 'Ingest' per compilare la wiki.", [], {})

    # Costruisci cronologia conversazione
    history_section = ""
    if session_history:
        history_lines = []
        for msg in session_history[-6:]:  # Ultimi 3 scambi
            role = "Utente" if msg.get("role") == "user" else "Assistente"
            history_lines.append(f"{role}: {msg.get('content', '')[:300]}")
        if history_lines:
            history_section = "\n\n=== CRONOLOGIA ===\n" + "\n".join(history_lines) + "\n=== FINE ===\n"

    schema = _read_file(get_wiki_schema_path())

    system_prompt = f"""Sei un assistente tecnico esperto specializzato in procedure di assemblaggio automotive.
Rispondi alle domande basandoti ESCLUSIVAMENTE sulla wiki fornita.

Regole:
- Usa informazioni dalla wiki, NON inventare
- Quando citi informazioni da una pagina immagine (wiki/images/), includi sempre il path dell'immagine nel formato `images/<nomefile>` — sarà mostrata come fonte cliccabile
- Cita le fonti (documento + pagina) quando possibile
- Se l'informazione non è nella wiki, dillo chiaramente
- Segui i [[link]] per trovare contesto aggiuntivo
- Sii diretto, conciso e professionale
- Usa elenchi puntati/numerati per procedure sequenziali"""

    user_prompt = f"""{history_section}
## Wiki Knowledge Base
{wiki_context}

## Domanda
{query}

## Risposta"""

    t0 = time.time()
    answer, usage = call_llm_text(config, system_prompt, user_prompt)
    llm_time_ms = (time.time() - t0) * 1000

    # Estrai le pagine wiki effettivamente caricate come contesto
    # Cerca i tag "=== FILE: wiki/{cat}/{name} ===" nel contesto
    context_pages = re.findall(r'=== FILE: wiki/(\S+/[^=\s]+\.md) ===', wiki_context)

    # Prova anche a trovare riferimenti ai file nell'answer (il LLM cita spesso il nome file)
    # es. "istruzioni-montaggio-automotive-v2.md" o "(Fonte: xxx.md)"
    answer_refs = re.findall(r'([\w-]+\.md)', answer)

    # Costruisci mappa filename -> path completo per le pagine in contesto
    path_by_name = {p.split('/')[-1]: p for p in context_pages}

    # Unisci: pagine del contesto il cui filename appare anche nell'answer (citate)
    # Se nessuna pagina è citata esplicitamente, mostra tutte quelle usate come contesto
    cited = [p for p in context_pages if p.split('/')[-1] in answer_refs]
    pages_used = cited if cited else context_pages

    return answer, pages_used, usage, wiki_context


# ============================================================
# LINT — Audit della wiki
# ============================================================

def lint_wiki(model_config: dict = None) -> dict:
    """
    Esegue un audit della wiki: trova contraddizioni, orfani, gap.

    Returns:
        dict con risultati lint {issues[], suggestions[], stats}
    """
    config = model_config or MODEL_PROMPT
    _ensure_wiki_dirs()

    all_wiki_content = _load_all_wiki_content(max_pages=50000)
    schema = _read_file(get_wiki_schema_path())

    if not all_wiki_content.strip():
        return {
            "success": True,
            "issues": [],
            "suggestions": ["La wiki è vuota. Carica documenti e usa Ingest."],
            "stats": get_wiki_status(),
        }

    system_prompt = f"""Sei un wiki auditor esperto. Analizza la wiki e trova problemi.

{schema}

IMPORTANTE: Rispondi SOLO con un JSON valido."""

    user_prompt = f"""## Operazione: LINT (Audit Wiki)

### Contenuto completo della wiki
{all_wiki_content}

### Istruzioni
Analizza tutta la wiki e rispondi con questo JSON:

```json
{{
  "issues": [
    {{
      "type": "orphan" | "broken_link" | "contradiction" | "missing_page" | "incomplete",
      "severity": "low" | "medium" | "high",
      "page": "categoria/nome-file.md",
      "description": "Descrizione del problema"
    }}
  ],
  "suggestions": [
    "Suggerimento testuale per migliorare la wiki"
  ],
  "health_score": 85
}}
```

Cerca:
1. Pagine orfane (nessun link in entrata)
2. Link rotti ([[riferimento]] a pagine inesistenti)
3. Contraddizioni tra pagine
4. Concetti menzionati ma senza pagina propria
5. Pagine incomplete o con poco contenuto"""

    t0 = time.time()
    response_text, usage = call_llm_text(config, system_prompt, user_prompt)
    llm_time_ms = (time.time() - t0) * 1000

    # Parse risposta
    result = _parse_json_response(response_text)

    _append_log(f"**LINT** — Health score: {result.get('health_score', '?')}%, "
                f"Issues: {len(result.get('issues', []))}")

    return {
        "success": True,
        "issues": result.get("issues", []),
        "suggestions": result.get("suggestions", []),
        "health_score": result.get("health_score", 0),
        "stats": get_wiki_status(),
        "llm_time_ms": round(llm_time_ms, 2),
        "usage": usage,
    }


def _parse_json_response(text: str) -> dict:
    """Estrae JSON dalla risposta LLM."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {"issues": [], "suggestions": ["Errore nel parsing della risposta LLM."], "health_score": 0}


# ============================================================
# FIX — Applica le correzioni suggerite dal Lint
# ============================================================

def fix_wiki(issues: list, suggestions: list, model_config: dict = None) -> dict:
    """
    Applica automaticamente le correzioni suggerite dall'audit Lint.

    Prende issues e suggestions dal risultato di lint_wiki() e usa l'LLM per:
    - Correggere link rotti
    - Completare pagine incomplete
    - Risolvere contraddizioni
    - Creare pagine mancanti segnalate
    - Applicare i suggerimenti generali

    Returns:
        dict con {success, pages_created, pages_updated, fixes_applied, log_entry}
    """
    if not issues and not suggestions:
        return {
            "success": True,
            "pages_created": 0,
            "pages_updated": 0,
            "fixes_applied": 0,
            "log_entry": "Nessun problema da correggere.",
        }

    config = model_config or MODEL_PROMPT
    # Aumenta token per risposta più lunga
    config = dict(config)
    if config.get("provider") == "openai":
        config["max_completion_tokens"] = max(config.get("max_completion_tokens", 0), 8192)
    else:
        config["max_tokens"] = max(config.get("max_tokens", 0), 8192)

    _ensure_wiki_dirs()

    existing_wiki = _load_all_wiki_content()
    schema = _read_file(get_wiki_schema_path())

    # Formatta issues e suggestions per il prompt
    issues_text = "\n".join(
        f"- [{i.get('severity','?').upper()}] [{i.get('type','?')}] {i.get('page','?')}: {i.get('description','')}"
        for i in issues
    ) or "Nessun issue specifico."

    suggestions_text = "\n".join(f"- {s}" for s in suggestions) or "Nessun suggerimento specifico."

    system_prompt = f"""Sei un wiki maintainer esperto. Il tuo compito è correggere i problemi
trovati durante un audit della wiki e applicare i suggerimenti di miglioramento.

{schema}

IMPORTANTE:
- Rispondi SOLO con un JSON valido. Nessun testo prima o dopo.
- Aggiorna SOLO le pagine che richiedono correzioni, non riscrivere l'intera wiki.
- Per i link rotti: correggi il [[link]] o rimuovilo se la pagina non esiste.
- Per le pagine incomplete: aggiungi contenuto rilevante basandoti sul contesto della wiki.
- Per le contraddizioni: scegli la versione più accurata e aggiorna entrambe le pagine.
- Per le pagine mancanti: creale con contenuto appropriato basandoti su quanto menzione la wiki.
- Il contenuto deve essere CONCISO (max 200 parole per pagina nuova)."""

    user_prompt = f"""## Operazione: FIX (Manutenzione Wiki)

### Problemi trovati dall'audit
{issues_text}

### Suggerimenti di miglioramento
{suggestions_text}

### Wiki attuale (completa)
{existing_wiki}

### Istruzioni
Correggi TUTTI i problemi elencati e applica i suggerimenti. Rispondi con questo JSON:

```json
{{
  "pages": [
    {{
      "action": "create" | "update",
      "category": "concepts" | "procedures" | "components" | "sources" | "images",
      "filename": "nome-file.md",
      "content": "# Titolo\\n\\nContenuto corretto/aggiornato..."
    }}
  ],
  "fixes_applied": [
    "Breve descrizione di ogni correzione applicata"
  ],
  "log_entry": "Riepilogo delle correzioni effettuate"
}}
```

Correggi ogni issue in modo mirato. Non modificare pagine che non hanno problemi."""

    t0 = time.time()
    response_text, usage = call_llm_text(config, system_prompt, user_prompt)
    llm_time_ms = (time.time() - t0) * 1000

    result = _parse_json_response(response_text)

    wiki_dir = get_wiki_dir()
    pages_created = 0
    pages_updated = 0

    for page in result.get("pages", []):
        cat = page.get("category", "concepts")
        if cat not in WIKI_CATEGORIES:
            cat = "concepts"
        fname = page.get("filename", "untitled.md")
        if not fname.endswith(".md"):
            fname += ".md"
        # Sanitize per sicurezza
        fname = os.path.basename(fname)

        filepath = os.path.join(wiki_dir, cat, fname)
        existed = os.path.exists(filepath)
        _write_file(filepath, page.get("content", ""))

        if existed:
            pages_updated += 1
        else:
            pages_created += 1

    fixes_applied = result.get("fixes_applied", [])
    log_entry = result.get("log_entry", f"Fix: {pages_created} create, {pages_updated} aggiornate")

    _append_log(
        f"**FIX** — {log_entry} "
        f"({pages_created} create, {pages_updated} aggiornate, {len(fixes_applied)} correzioni)"
    )

    return {
        "success": True,
        "pages_created": pages_created,
        "pages_updated": pages_updated,
        "fixes_applied": len(fixes_applied),
        "fixes_detail": fixes_applied,
        "log_entry": log_entry,
        "llm_time_ms": round(llm_time_ms, 2),
        "usage": usage,
    }


# ============================================================
# LOG
# ============================================================

def _append_log(entry: str):
    """Aggiunge un'entry al log della wiki."""
    log_path = get_wiki_log_path()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"\n## {timestamp}\n{entry}\n"

    existing = _read_file(log_path)
    _write_file(log_path, existing + log_line)


def get_wiki_log() -> str:
    """Restituisce il contenuto del log della wiki."""
    return _read_file(get_wiki_log_path())


# ============================================================
# RESET
# ============================================================

def reset_wiki():
    """Resetta la wiki: elimina tutte le pagine generate (mantiene schema)."""
    wiki_dir = get_wiki_dir()
    _ensure_wiki_dirs()

    deleted = 0
    for sub in WIKI_CATEGORIES:
        sub_dir = os.path.join(wiki_dir, sub)
        for fp in glob.glob(os.path.join(sub_dir, "*.md")):
            os.remove(fp)
            deleted += 1

    # Reset index e log
    _write_file(get_wiki_index_path(),
                "# Wiki Index\n\n> Knowledge base vuota. Usa Ingest per compilare.\n\n"
                "*Ultimo aggiornamento: " + datetime.now().strftime("%Y-%m-%d %H:%M") + "*\n")
    _write_file(get_wiki_log_path(),
                "# Wiki Log\n\n> Registro cronologico.\n\n---\n")

    _append_log(f"**RESET** — Eliminate {deleted} pagine. Wiki ripristinata.")

    return {"success": True, "pages_deleted": deleted}
