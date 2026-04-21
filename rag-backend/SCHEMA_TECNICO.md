# Relazione Tecnica: Sistema RAG Multimodale con Contextual Retrieval

## Executive Summary

Questo documento descrive l'architettura e l'implementazione di un sistema **RAG (Retrieval-Augmented Generation) Multimodale** che combina:
- **Contextual Retrieval** per chunk di testo (tecnica Anthropic 2024)
- **Analisi immagini** con Claude Opus 4 Vision
- **Ricerca semantica** con FAISS e Sentence Transformers

Il sistema è progettato per estrarre e ricercare informazioni da documenti tecnici contenenti sia testo che immagini (schemi, diagrammi, planimetrie).

---

## 1. Architettura del Sistema

### 1.1 Overview Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FASE DI INDICIZZAZIONE                             │
│                              (build_index.py)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────┐    ┌────────────┐ │
│  │ DOCUMENTI│───▶│ ESTRAZIONE   │───▶│ CONTEXTUAL      │───▶│ EMBEDDING  │ │
│  │ (PDF,TXT)│    │ TESTO+IMG    │    │ RETRIEVAL       │    │ GENERATION │ │
│  └──────────┘    └──────────────┘    └─────────────────┘    └────────────┘ │
│                                                                      │       │
│                                                                      ▼       │
│                                                              ┌────────────┐ │
│                                                              │ FAISS      │ │
│                                                              │ INDEX      │ │
│                                                              └────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                            FASE DI QUERY                                     │
│                            (rag_query.py)                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────┐    ┌────────────┐ │
│  │ DOMANDA  │───▶│ EMBEDDING    │───▶│ FAISS SEARCH    │───▶│ RE-RANKING │ │
│  │ UTENTE   │    │ QUERY        │    │ + RETRIEVAL     │    │ IMMAGINI   │ │
│  └──────────┘    └──────────────┘    └─────────────────┘    └────────────┘ │
│                                                                      │       │
│                                                                      ▼       │
│                       ┌────────────┐    ┌─────────────────────────────────┐ │
│                       │ RISPOSTA   │◀───│ LLM (Claude Opus)               │ │
│                       │ FINALE     │    │ con contesto testo + immagini   │ │
│                       └────────────┘    └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Flusso Dettagliato: Injection → Embedding → Salvataggio

#### STEP 1: Caricamento Documenti

```python
# Formati supportati
SUPPORTED_FORMATS = ['.pdf', '.txt', '.md', '.docx', '.xlsx', '.xls']

# Output: lista di documenti con path, testo estratto, full_path
docs = [
    {
        'path': 'documento.pdf',
        'text': 'contenuto estratto...',
        'full_path': '/path/to/documento.pdf'
    }
]
```

#### STEP 2: Chunking del Testo

Il testo viene suddiviso in chunk con parametri configurabili:

| Parametro | Valore Default | Descrizione |
|-----------|----------------|-------------|
| `CHUNK_SIZE` | 800 caratteri | Dimensione massima chunk |
| `CHUNK_OVERLAP` | 200 caratteri | Sovrapposizione tra chunk consecutivi |

```python
def chunk_text(text, chunk_size=800, overlap=200):
    """
    - Split su fine frase (. ? !)
    - Overlap per contesto continuo
    - Preserva integrità semantica
    """
```

#### STEP 3: Contextual Retrieval (Testo)

Tecnica innovativa di Anthropic (2024) che arricchisce ogni chunk con contesto LLM:

```
PRIMA (chunk grezzo):
"Il filtro aria protegge il motore da polvere e impurità. 
 Sostituzione ogni 30.000 km."

DOPO (chunk contestualizzato):
"CONTEXT: Section 1.2 on air filter replacement in the car 
maintenance manual. Covers replacement interval and protective 
function of the air filter component.

CONTENT: Il filtro aria protegge il motore da polvere e impurità. 
Sostituzione ogni 30.000 km."
```

**Vantaggi del Contextual Retrieval:**
- Migliora la ricercabilità semantic del chunk
- Aggiunge metadati impliciti (sezione, argomento)
- Risolve ambiguità di riferimenti interni

#### STEP 4: Estrazione Immagini da PDF

Pipeline ibrida per catturare tutti i tipi di immagini:

```
┌────────────────────────────────────────────────────────────────┐
│                   ESTRAZIONE IMMAGINI                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. PyMuPDF (fitz)         2. OpenCV Layout Detection          │
│  ─────────────────         ──────────────────────────          │
│  • Immagini bitmap         • Render pagina 300 DPI             │
│  • JPG/PNG embedded        • Canny edge detection              │
│  • Foto, screenshot        • Morphology dilate                 │
│                            • Contour detection                  │
│                            • Crop regioni                       │
│              │                        │                         │
│              └────────────┬───────────┘                         │
│                           ▼                                     │
│                    FILTRO DIMENSIONE                            │
│                    (min 100px)                                  │
│                           │                                     │
│                           ▼                                     │
│                   SALVATAGGIO IN:                               │
│              output/extracted_images/                           │
└────────────────────────────────────────────────────────────────┘
```

#### STEP 5: Analisi Immagini con Claude Opus Vision

Ogni immagine viene analizzata con contesto della pagina:

```python
def analyze_image_with_context(image_path, page_num, source_doc, page_text=None):
    """
    Input:
    - image_path: path immagine estratta
    - page_num: pagina di origine
    - source_doc: nome documento
    - page_text: testo della pagina (contesto)
    - prompt di sistema dettagliato
    
    Output:
    - Descrizione tecnica dettagliata dell'immagine
    """
```

**Prompt di analisi immagine:**
```
Sei un esperto ingegnere che analizza disegni tecnici...

Analizza questa immagine dalla pagina X del documento Y.

CONTESTO DELLA PAGINA X:
[testo della pagina - max 1500 caratteri]

Focalizzati su:
1. Tipo di contenuto (schema, diagramma, grafico, planimetria...)
2. ARGOMENTO SPECIFICO dal contesto
3. Componenti principali e identificatori
4. Valori numerici, misure, specifiche tecniche
5. Connessioni, relazioni, flussi
6. Testo presente nell'immagine
7. Scopo tecnico del disegno
```

#### STEP 6: Calcolo Embeddings

```python
# Modello: Sentence Transformers
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# Dimensione vettore: 384
# Embedding calcolato su: text_for_embedding (testo contestualizzato)

embeddings = model.encode(texts, convert_to_numpy=True)
embeddings = embeddings / np.linalg.norm(embeddings)  # L2 normalization
```

#### STEP 7: Costruzione Index FAISS

```python
# Tipo index: IndexFlatIP (Inner Product per cosine similarity)
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# Salvataggio:
# - output/docs_index_multimodal_contextual.faiss
# - output/chunks_multimodal_contextual.jsonl
# - output/metadata_multimodal_contextual.json
```

### 1.3 Struttura Chunk Salvato

```json
{
    "chunk_id": 0,
    "type": "text",           // "text" | "image" | "table"
    "source": "documento.pdf",
    "text_original": "contenuto originale...",
    "text_for_embedding": "CONTEXT: ... CONTENT: ...",
    "page": 1,
    "image_path": null        // presente solo per type="image"
}
```

---

## 2. Logica di Validazione e Rilevanza Immagini

### 2.1 Sistema di Scoring Multi-Criterio

Il sistema utilizza una precisa logica per determinare quali immagini includere nella risposta:

```
┌───────────────────────────────────────────────────────────────────────────┐
│                    ALGORITMO SELEZIONE IMMAGINI                            │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   Query Utente ──────┬──────────────────────────────────────────────────▶ │
│                      │                                                     │
│                      ▼                                                     │
│              ┌───────────────┐                                             │
│              │ FAISS Search  │──▶ Similarity Score (0.0 - 1.0)            │
│              └───────────────┘                                             │
│                      │                                                     │
│                      ▼                                                     │
│              ┌───────────────┐                                             │
│              │ Keyword       │──▶ Keyword Overlap (0.0 - 1.0)             │
│              │ Extraction    │                                             │
│              └───────────────┘                                             │
│                      │                                                     │
│                      ▼                                                     │
│              ┌───────────────────────────────────────┐                     │
│              │     DECISION MATRIX                    │                     │
│              │  ┌──────────────────────────────────┐ │                     │
│              │  │ Case 1: Score ≥ 0.55 AND         │ │                     │
│              │  │         Keyword ≥ 0.20  ──▶ ✅    │ │                     │
│              │  ├──────────────────────────────────┤ │                     │
│              │  │ Case 2: Score ≥ 0.45 AND         │ │                     │
│              │  │         Keyword ≥ 0.30  ──▶ ✅    │ │                     │
│              │  ├──────────────────────────────────┤ │                     │
│              │  │ Case 3: Score ≥ 0.65   ──▶ ✅    │ │                     │
│              │  │         (sempre inclusa)         │ │                     │
│              │  ├──────────────────────────────────┤ │                     │
│              │  │ Case 4: Keyword ≥ 0.75 ──▶ ✅    │ │                     │
│              │  │         (alta pertinenza kw)     │ │                     │
│              │  └──────────────────────────────────┘ │                     │
│              └───────────────────────────────────────┘                     │
│                                                                            │
└───────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Parametri di Configurazione (config.py)

I valori del DECISION MATRIX sono parametrizzabili in base al livello di correlazione che si vuole ottenere.

| Parametro | Valore attuale | Descrizione |
|-----------|--------|-------------|
| `IMAGE_HIGH_SCORE_THRESHOLD` | 0.65 | Score sopra cui l'immagine è sempre inclusa |
| `IMAGE_SCORE_THRESHOLD` | 0.55 | Score medio per considerare un'immagine |
| `IMAGE_KEYWORD_OVERLAP_MIN` | 0.20 | Overlap minimo keyword per immagini con score medio |
| `IMAGE_SCORE_MIN_WITH_KEYWORD` | 0.45 | Score minimo se c'è alto overlap keyword |
| `IMAGE_KEYWORD_OVERLAP_MAX` | 0.30 | Overlap keyword per compensare lo score minimo |
| `IMAGE_KEYWORD_BOOST_THRESHOLD` | 0.75 | Overlap keyword sopra cui l'immagine è sempre inclusa |

### 2.3 Calcolo Keyword Overlap

```python
def calculate_keyword_overlap(query, image_description):
    """
    1. Tokenizzazione query (rimozione stopwords)
    2. Tokenizzazione descrizione immagine
    3. Intersezione set di parole
    4. Overlap = |intersezione| / |query_words|
    """
    stopwords = {'il', 'lo', 'la', 'di', 'da', 'a', 'in', 'su', ...}
    query_words = set(w for w in query.lower().split() 
                      if w not in stopwords and len(w) > 2)
    desc_words = set(w for w in description.lower().split() 
                     if len(w) > 2)
    
    overlap = len(query_words & desc_words) / max(len(query_words), 1)
    return overlap
```

### 2.4 Post-Filtering nella Risposta: Due Filtri, Due Scopi

Il sistema implementa **due filtri distinti** che operano in momenti diversi della pipeline e con obiettivi complementari:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PIPELINE DUE FILTRI IMMAGINI                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  QUERY: "Come si cambia l'olio?"                                            │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  1° FILTRO: retrieve() - PRIMA dell'LLM                             │    │
│  │  ───────────────────────────────────────                            │    │
│  │  Scopo: Selezionare immagini DA PASSARE ALL'LLM come contesto       │    │
│  │                                                                      │    │
│  │  Input:  30 chunk totali (20 testo + 10 immagini)                   │    │
│  │  Output: 6 chunk per l'LLM (3 testo + 3 immagini)                   │    │
│  │                                                                      │    │
│  │  Criteri: Score + Keyword Overlap (vedi sezione 2.1)                │    │
│  │                                                                      │    │
│  │  Passa: img_olio.png ✅, img_filtro.png ✅, img_batteria.png ✅     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  LLM genera risposta con il contesto ricevuto                       │    │
│  │  ────────────────────────────────────────────                       │    │
│  │  "Per cambiare l'olio: 1) sollevare auto 2) svitare tappo...        │    │
│  │   Come mostrato nel DISEGNO TECNICO della pagina 2..."              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  2° FILTRO: post-risposta - DOPO l'LLM                              │    │
│  │  ─────────────────────────────────────────                          │    │
│  │  Scopo: Decidere quali immagini MOSTRARE ALL'UTENTE nel risultato   │    │
│  │                                                                      │    │
│  │  Verifica: L'LLM ha EFFETTIVAMENTE USATO queste immagini?           │    │
│  │                                                                      │    │
│  │  img_olio.png     → LLM cita "disegno pagina 2"  ✅ MOSTRA          │    │
│  │  img_filtro.png   → score 0.72, keyword match    ✅ MOSTRA          │    │
│  │  img_batteria.png → non citata, score 0.58       ❌ NON MOSTRARE    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│         │                                                                    │
│         ▼                                                                    │
│  OUTPUT: Risposta + solo immagini realmente pertinenti                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Perché servono entrambi i filtri?

| Aspetto | 1° Filtro (retrieve) | 2° Filtro (post-risposta) |
|---------|---------------------|---------------------------|
| **Quando** | Prima della chiamata LLM | Dopo la generazione risposta |
| **Cosa decide** | Immagini nel **prompt** | Immagini **visualizzate** |
| **Criteri** | Score semantico + keyword | Citazione nella risposta + score |
| **Obiettivo** | Dare contesto all'LLM | Mostrare solo il rilevante all'utente |

**Il secondo filtro NON invalida il primo** - aggiunge un layer di **intelligenza contestuale**:
- Un'immagine può essere passata all'LLM (1° filtro ok) ma l'LLM decide di non usarla → non ha senso mostrarla
- Il 2° filtro può verificare se l'LLM ha citato l'immagine nella risposta generata

#### Codice del 2° Filtro

```python
# Criteri per mostrare immagine nel risultato finale:
is_relevant = (
    (idx < 3 and score >= 0.65) or           # Top 3 risultati con score alto
    is_mentioned_in_answer(answer, image) or  # Citata esplicitamente nella risposta
    score >= 0.75 or                          # Score molto alto (sempre rilevante)
    (keyword_overlap >= 0.5 and score >= 0.70)  # Buon match keyword + score
)

# Verifica menzione nella risposta
is_mentioned = (
    f"pagina {page}" in answer_lower or      # Riferimento alla pagina
    f"pag. {page}" in answer_lower or
    "immagine" in answer_lower or            # Parole chiave generiche
    "figura" in answer_lower or
    "disegno" in answer_lower or
    source.lower() in answer_lower           # Nome documento citato
)
```

---

## 3. Logging e Persistenza Dati (Proposta Database)

### 3.1 Schema Database Proposto

Per un sistema completo, si consiglia la seguente struttura relazionale + vettoriale:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SCHEMA DATABASE                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐    │
│  │   DOCUMENTS       │     │     CHUNKS        │     │    EMBEDDINGS    │    │
│  ├──────────────────┤     ├──────────────────┤     ├──────────────────┤    │
│  │ id (PK)          │     │ id (PK)          │     │ chunk_id (FK)    │    │
│  │ filename         │────▶│ document_id (FK) │────▶│ vector (384 dim) │    │
│  │ file_path        │     │ chunk_index      │     │ created_at       │    │
│  │ file_type        │     │ type             │     └──────────────────┘    │
│  │ total_pages      │     │ text_original    │                              │
│  │ upload_date      │     │ text_context     │     ┌──────────────────┐    │
│  │ processed_date   │     │ page_number      │     │    IMAGES        │    │
│  │ status           │     │ image_path       │     ├──────────────────┤    │
│  │ metadata_json    │     │ created_at       │     │ chunk_id (FK)    │    │
│  └──────────────────┘     └──────────────────┘     │ file_path        │    │
│                                                     │ width            │    │
│  ┌──────────────────┐     ┌──────────────────┐     │ height           │    │
│  │   SESSIONS       │     │     QUERIES       │     │ analysis_text    │    │
│  ├──────────────────┤     ├──────────────────┤     │ bbox_json        │    │
│  │ id (PK)          │     │ id (PK)          │     └──────────────────┘    │
│  │ user_id          │────▶│ session_id (FK)  │                              │
│  │ started_at       │     │ query_text       │                              │
│  │ ended_at         │     │ query_embedding  │                              │
│  │ conversation_json│     │ timestamp        │                              │
│  └──────────────────┘     │ response_text    │                              │
│                           │ chunks_used      │                              │
│                           │ images_shown     │                              │
│                           │ latency_ms       │                              │
│                           │ feedback_score   │                              │
│                           └──────────────────┘                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Tipi di Log da Salvare

#### A. Log di Indicizzazione (Build-Time)

```json
{
    "event_type": "document_indexed",
    "timestamp": "2026-02-20T10:30:00Z",
    "document": {
        "id": "doc_001",
        "filename": "manuale_manutenzione.pdf",
        "pages": 50,
        "processing_time_ms": 45000
    },
    "chunks": {
        "total": 120,
        "text": 95,
        "images": 25
    },
    "images_extracted": [
        {
            "path": "output/extracted_images/doc_p1_bitmap1.png",
            "page": 1,
            "size": [800, 600],
            "type": "bitmap"
        }
    ],
    "llm_calls": {
        "contextualization": 12,
        "image_analysis": 25,
        "total_tokens": 150000
    }
}
```

#### B. Log di Query (Run-Time)

```json
{
    "event_type": "query_executed",
    "timestamp": "2026-02-20T14:25:00Z",
    "session_id": "sess_abc123",
    "query": {
        "id": "qry_xyz789",
        "text": "Come si cambia l'olio motore?",
        "embedding_time_ms": 15
    },
    "retrieval": {
        "search_time_ms": 8,
        "chunks_retrieved": 5,
        "chunks_text": 3,
        "chunks_images": 2,
        "top_scores": [0.89, 0.85, 0.78, 0.72, 0.68]
    },
    "response": {
        "llm_model": "claude-opus-4-6",
        "tokens_input": 2500,
        "tokens_output": 450,
        "generation_time_ms": 3200,
        "text": "Per cambiare l'olio motore...",
        "images_shown": [{ "name": "p2_bitmap1.png", "score": "56%", "keyword": "65%" }]
    },
    "feedback": {
        "user_rating": null,
        "was_helpful": null
    }
}
```

#### C. Log di Performance e Monitoring

```json
{
    "event_type": "system_metrics",
    "timestamp": "2026-02-20T15:00:00Z",
    "metrics": {
        "index_size_mb": 125.5,
        "total_vectors": 5420,
        "avg_query_latency_ms": 180,
        "llm_errors_24h": 2,
        "cache_hit_rate": 0.45
    }
}
```

### 3.3 Tecnologie Consigliate (AWS)

| Componente | Tecnologia AWS |
|------------|----------------|
| **DB Relazionale** | Amazon RDS for PostgreSQL |
| **DB Vettoriale** | pgvector su RDS / Amazon OpenSearch |
| **Log Storage** | Amazon CloudWatch Logs |
| **LLM Hosting** | Amazon Bedrock |

---

## 4. Gestione Memoria Conversazionale

### 4.1 Architettura Conversazione

Per portare in memoria la conversazione precedente, si propone il seguente approccio:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CONVERSATION MEMORY ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   UTENTE                                                                     │
│      │                                                                       │
│      ▼                                                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     SESSION MANAGER                                   │   │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │   │
│  │  │ Session Store   │    │ History Buffer  │    │ Context Window  │  │   │
│  │  │ (Redis/DB)      │    │ (Last N turns)  │    │ (Token Limit)   │  │   │
│  │  └─────────────────┘    └─────────────────┘    └─────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                         │                                    │
│                                         ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      PROMPT BUILDER                                   │   │
│  │                                                                       │   │
│  │   [SYSTEM PROMPT]                                                    │   │
│  │   + [CONVERSATION HISTORY]  ◀── Ultime N interazioni                │   │
│  │   + [RETRIEVED CONTEXT]     ◀── Chunk da RAG                        │   │
│  │   + [CURRENT QUERY]                                                  │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                         │                                    │
│                                         ▼                                    │
│                                    ┌─────────┐                              │
│                                    │   LLM   │                              │
│                                    └─────────┘                              │
│                                         │                                    │
│                                         ▼                                    │
│                                   RISPOSTA                                   │
│                                         │                                    │
│                                         ▼                                    │
│                          SALVA IN SESSION STORE                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Architettura AWS Dettagliata

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AWS PRODUCTION ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────┐         ┌─────────────┐         ┌─────────────┐           │
│   │   Route 53  │────────▶│     ALB     │────────▶│    EKS      │           │
│   │   (DNS)     │         │ (Load Bal.) │         │  Cluster    │           │
│   └─────────────┘         └─────────────┘         └──────┬──────┘           │
│                                                          │                   │
│                           ┌──────────────────────────────┼───────────────┐  │
│                           │           VPC                │               │  │
│                           │    ┌─────────────────────────┴────────────┐ │  │
│                           │    │         Private Subnets              │ │  │
│                           │    │  ┌─────────┐  ┌─────────┐  ┌──────┐ │ │  │
│                           │    │  │ RAG API │  │ Worker  │  │Redis │ │ │  │
│                           │    │  │  Pods   │  │  Pods   │  │Cache │ │ │  │
│                           │    │  └────┬────┘  └────┬────┘  └──────┘ │ │  │
│                           │    └───────┼────────────┼────────────────┘ │  │
│                           │            │            │                   │  │
│                           │    ┌───────┴────────────┴───────────────┐  │  │
│                           │    │              Data Layer            │  │  │
│                           │    │  ┌──────────┐     ┌─────────────┐  │  │  │
│                           │    │  │   RDS    │     │     S3      │  │  │  │
│                           │    │  │PostgreSQL│     │   Bucket    │  │  │  │
│                           │    │  │+pgvector │     │  (docs/img) │  │  │  │
│                           │    │  └──────────┘     └─────────────┘  │  │  │
│                           │    └────────────────────────────────────┘  │  │
│                           └────────────────────────────────────────────┘  │
│                                                                           │
│   ┌─────────────────────────────────────────────────────────────────────┐ │
│   │                      AWS AI Services                                │ │
│   │  ┌──────────────┐    ┌──────────────┐    ┌───────────────┐          │ │
│   │  │   Bedrock    │    │  Textract    │    │  Rekognition  │          │ │
│   │  │   (Claude)   │    │ (OCR boost)  │    │ (img analysis)│          │ │
│   │  └──────────────┘    └──────────────┘    └───────────────┘          │ │
│   └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Appendice A: Configurazione Parametri

```python
# config.py - Parametri chiave

# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
BATCH_SIZE = 10

# Immagini
MIN_IMAGE_SIZE = 100
USE_LAYOUT_DETECTION = True

# Retrieval
TOP_K_TEXT = 3
TOP_K_IMAGES = 3
MIN_K_IMAGES = 0
SEARCH_MULTIPLIER = 5

# Soglie immagini
IMAGE_HIGH_SCORE_THRESHOLD = 0.65
IMAGE_SCORE_THRESHOLD = 0.55
IMAGE_KEYWORD_OVERLAP_MIN = 0.20
IMAGE_SCORE_MIN_WITH_KEYWORD = 0.45
IMAGE_KEYWORD_OVERLAP_MAX = 0.30
IMAGE_KEYWORD_BOOST_THRESHOLD = 0.75

# Embedding
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
```

---

*Documento generato il 20 Febbraio 2026*
*Versione: 1.0*
