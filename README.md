# RAG Multimodale + LLM Wiki

## Cos'è

Sistema che combina un pipeline **RAG multimodale** (testo + immagini da PDF) con un layer di **LLM Wiki**: una knowledge base strutturata a pagine Markdown generata e aggiornata automaticamente da un LLM a partire dai documenti indicizzati.

L'LLM Wiki non è un semplice indice vettoriale: è una raccolta di pagine Markdown scritte dal modello, interconnesse tramite wikilink (`[[nome-pagina]]`), che il modello legge come contesto pre-sintetizzato prima di rispondere alle domande dell'utente.

---

## Funzionalità

### Pipeline RAG (indicizzazione)

- Estrazione testo da PDF, DOCX, TXT, XLSX tramite Azure Document Intelligence (layout + tabelle) o PyPDF
- Estrazione immagini bitmap e vettoriali (OpenCV layout detection)
- Analisi visiva di ogni immagine con un modello LLM multimodale
- Chunking + Contextual Retrieval: ogni chunk riceve un contesto generato dal LLM
- Embedding con Sentence Transformers (`all-MiniLM-L6-v2`) e indice FAISS locale
- Supporto opzionale per Azure AI Search come vector store cloud

### LLM Wiki

| Operazione | Descrizione |
|---|---|
| **Ingest documento** | Il LLM legge il documento indicizzato e genera/aggiorna pagine wiki strutturate |
| **Ingest immagini** | Ogni immagine viene analizzata con LLM vision e ottiene una pagina wiki in `images/` |
| **Learn** | L'utente inserisce testo libero; il LLM lo integra nella wiki come nuove pagine o aggiornamenti |
| **Lint** | Analisi stato di salute della wiki con estrazioni di suggerimenti migliorativi della stessa |
| **Query** | Il LLM legge le pagine wiki pre-sintetizzate piè rilevanti per rispondere alla domanda |
| **Graph** | Visualizzazione a grafo degli inter-link tra pagine wiki |

### Web App

Frontend Next.js con:

- Chat RAG con citazione delle fonti (chunk testuali + immagini)
- Sidebar documenti con upload e build indice
- Viewer wiki con navigazione per categoria e apertura pagine
- Grafo interattivo dei wikilink
- Log dell'indicizzazione in tempo reale
- Selezione del modello LLM per ogni conversazione

---

## Struttura della Wiki

```
wiki/
+-- index.md          # Indice principale con link a tutte le pagine
+-- log.md            # Log cronologico di tutte le operazioni
+-- schema.md         # Regole di formato e comportamento per il LLM
+-- sources/          # Sommario di ogni documento ingerito (1 pagina/doc)
+-- concepts/         # Concetti tecnici (coppie di serraggio, materiali, standard...)
+-- procedures/       # Procedure operative step-by-step
+-- components/       # Componenti e Part Numbers
+-- images/           # Pagine generate per ogni immagine analizzata
```

Ogni pagina segue questo template:

```markdown
# Titolo

> Breve descrizione in una riga.

## Dettagli
Contenuto conciso (max 150-200 parole, elenchi puntati preferiti).

## Riferimenti Visivi
`images/nomefile.png`

## Collegamenti
- [[pagina-correlata]]

## Fonti
- Documento: nomefile.pdf, Pagina: N
```

---

## Flusso delle Operazioni

### 1. Indicizzazione documento

```
PDF / DOCX / TXT
       ?
Azure Document Intelligence  --------------------------+
(estrazione testo + tabelle)                           
       ?                                               
Chunking (800 char, overlap 200)         Estrazione immagini
       ?                                               
Contextual Retrieval                                   
(LLM genera contesto per chunk)         Analisi vision LLM
       ?                                               
Sentence Transformers embedding  ?---------------------+
       ?
FAISS index  +  metadata.json  +  chunks.jsonl
```

### 2. Ingest Wiki da documento

```
Documento indicizzato
       ?
wiki_manager.ingest_document()
       +- Carica wiki esistente come contesto
       +- Invia documento + wiki al LLM
       +- LLM risponde con JSON: {pages, index_update, log_entry}
       +- Scrive pagine in wiki/sources/, concepts/, procedures/, components/
       +- Aggiorna wiki/index.md
       +- Appende a wiki/log.md
       ? (se ci sono immagini)
wiki_manager.ingest_images_to_wiki()
       +- Per ogni immagine: call_llm_with_image() ? descrizione tecnica
       +- Scrive pagina in wiki/images/{filename}.md
```

### 3. Query con contesto Wiki

```
Domanda utente
       ?
_find_relevant_pages(query)    ? keyword scoring su tutte le pagine wiki
       ?
Top-K pagine wiki  +  FAISS retrieval (chunk testuali + immagini)
       ?
LLM genera risposta
       ?
Risposta  +  fonti PDF  +  immagini correlate
```

---

## Configurazione (config.py)

Il file `config.py` contiene credenziali sensibili e **non viene versionato**. Copiare `config_example.py` come `config.py` e compilare i valori:

```bash
cp config_example.py config.py
```

### Parametri di processing

```python
CHUNK_SIZE = 800          # Dimensione chunk di testo (caratteri)
CHUNK_OVERLAP = 200       # Overlap tra chunk consecutivi
BATCH_SIZE = 25           # Chunk per chiamata LLM nella contestualizzazione
LLM_CONCURRENCY = 4       # Chiamate LLM in parallelo

MIN_IMAGE_SIZE = 100      # Dimensione minima immagini estratte (pixel)
USE_LAYOUT_DETECTION = True  # OpenCV per figure vettoriali

TOP_K_TEXT = 10           # Chunk testuali da recuperare per query
TOP_K_IMAGES = 3          # Immagini massime per query
```

### Modelli LLM (MODEL_PROVIDERS)

Dizionario con tutti i modelli disponibili. Ogni entry ha questi campi:

```python
"nome-modello": {
    'provider': 'openai' | 'anthropic' | 'mistral' | 'mistral-ocr',
    'deployment_name': 'nome-deployment-azure',
    'endpoint': '<<endpoint>>',
    'api_key': '<chiave>',
    'max_tokens': 4096,              # per Anthropic/Mistral
    'max_completion_tokens': 4096,   # per OpenAI
    'temperature': 0,
    'api_version': '2024-12-01-preview',  # solo per OpenAI
    'name': 'Nome visualizzato nella UI',
}
```

I parametri `DEFAULT_MODEL_NAME` e `DEFAULT_IMAGE_MODEL_NAME` selezionano rispettivamente il modello usato per la chat/ingest e per l'analisi visiva delle immagini.

### Azure Document Intelligence

```python
DOCUMENT_INTELLIGENCE = {
    'endpoint': 'https://<risorsa>.cognitiveservices.azure.com/',
    'api_key': '<chiave>',
    'api_version': '2024-02-29-preview',
    'model_id': 'prebuilt-layout'
}
```

### Azure AI Search (opzionale è vector store cloud)

```python
AZURE_SEARCH = {
    'endpoint': 'https://<risorsa>.search.windows.net',
    'api_key': '<chiave>',
    'index_name': 'rag-multimodal-index',
    'embedding_dimensions': 384,
}
```

### Azure AI Vision (opzionale è analisi immagini alternativa)

```python
AZURE_VISION = {
    'endpoint': 'https://<risorsa>.cognitiveservices.azure.com/',
    'api_key': '<chiave>',
    'api_version': '2024-02-01',
    'features': ['caption', 'denseCaptions', 'tags', 'read'],
    'language': 'it',
    'gender_neutral_caption': True
}
```

### Wiki

```python
WIKI_DIR = "wiki"
WIKI_MAX_CONTEXT_PAGES = 15     # Max pagine wiki nel contesto LLM per query
WIKI_INGEST_MAX_TOKENS = 8192   # Dimensione di ogni segmento di testo inviato al LLM durante l'ingest (in token ×4 = caratteri)
WIKI_INGEST_MAX_BATCHES = 12    # Max chiamate LLM per documento; per documenti molto grandi i segmenti vengono campionati uniformemente fino a questo limite
```

`WIKI_INGEST_MAX_BATCHES` evita run indefiniti su documenti enormi: se il testo suddiviso genera più segmenti del limite, vengono selezionati `WIKI_INGEST_MAX_BATCHES` segmenti posizionati a distanza uniforme lungo tutto il documento (dal primo all'ultimo), garantendo copertura dell'intero contenuto.

---

## Avviare Backend e Frontend

### Prerequisiti

- Python 3.10+
- Node.js 18+ con pnpm

### Backend (FastAPI)

```bash
cd rag-backend

# Prima installazione
pip install -r requirements.txt

# Copia e compila la configurazione
cp config_example.py config.py
# ... edita config.py con le tue credenziali ...

# Avvia il server
py api_server.py
```

Il backend si avvia su `http://localhost:8000`.

Endpoint principali:

| Metodo | Path | Descrizione |
|--------|------|-------------|
| `GET` | `/api/wiki/status` | Stato della wiki (totale pagine per categoria) |
| `GET` | `/api/wiki/pages` | Lista tutte le pagine wiki |
| `GET` | `/api/wiki/graph` | Nodi e archi per la visualizzazione a grafo |
| `GET` | `/api/wiki/pages/{category}/{filename}` | Contenuto di una singola pagina |
| `POST` | `/api/wiki/ingest` | Avvia ingest wiki dal documento giè indicizzato |
| `POST` | `/api/wiki/learn` | Integra testo libero nella wiki |
| `POST` | `/api/query` | Esegue una query RAG (con contesto wiki) |
| `POST` | `/api/documents/upload` | Carica un documento |
| `POST` | `/api/index/build` | Avvia build dell'indice FAISS |

### Frontend (Next.js)

```bash
cd rag-web-app

# Prima installazione
pnpm install

# Avvia in modalitè sviluppo
pnpm dev
```

Il frontend si avvia su `http://localhost:3000` e si connette al backend su `http://localhost:8000`.

---

## Struttura File Backend

```
rag-backend/
+-- api_server.py                         # FastAPI: tutti gli endpoint HTTP
+-- wiki_manager.py                       # LLM Wiki: ingest, query, learn, graph
+-- llm_client.py                         # Client unificato (OpenAI/Anthropic/Mistral)
+-- build_index.py                        # Pipeline RAG con PyPDF + OpenCV
+-- build_index_document_intelligence.py  # Pipeline RAG con Azure Document Intelligence
+-- document_intelligence_extractor.py    # Estrazione testo/tabelle con Azure DI
+-- azure_search_client.py                # Client per Azure AI Search
+-- rag_logger.py                         # Logger per log indicizzazione
+-- config.py                             # Configurazione (NON versionato)
+-- config_example.py                     # Template configurazione senza credenziali
+-- requirements.txt
+-- docs/                                 # Documenti PDF/DOCX da indicizzare
+-- output/                               # Indice FAISS, metadata, chunk, immagini
+-- wiki/                                 # Knowledge base wiki (pagine Markdown)
```
