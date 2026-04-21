# 🚀 RAG Multimodale + Contextual Retrieval

## 🎯 Cos'è

Sistema RAG avanzato che estrae e comprende informazioni da **testo** e **immagini** dei PDF.

**Caratteristiche principali:**
- 📄 Contextual Retrieval per testo (Anthropic 2024)
- 🖼️ Estrazione automatica immagini (bitmap + grafici vettoriali)
- 🤖 Analisi immagini con Claude Opus 4-6 / Azure AI Vision
- 🎯 Selezione intelligente delle immagini rilevanti
- 📊 Supporto tabelle con Azure Document Intelligence

---

## 🔄 Pipeline di Build - Step-by-Step

### 📘 1. Build Standard (`build_index.py`)

Usa **PyPDF + OpenCV + Claude Opus** per l'indicizzazione.

```
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 1: Caricamento Documenti                                       │
├─────────────────────────────────────────────────────────────────────┤
│ • Legge file da cartella ./docs                                     │
│ • Supporta: PDF, TXT, MD, DOCX, XLSX, XLS                          │
│ • PyPDF per estrazione testo da PDF                                 │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 2: Chunking Testo                                              │
├─────────────────────────────────────────────────────────────────────┤
│ • Divide testo in chunk (default: 800 caratteri)                    │
│ • Overlap tra chunk (default: 200 caratteri)                        │
│ • Split su fine frase (. ? !)                                       │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 3: Contextual Retrieval (Testo)                                │
├─────────────────────────────────────────────────────────────────────┤
│ • Claude Opus genera contesto per ogni chunk                        │
│ • Batch processing (default: 10 chunk per chiamata)                 │
│ • Output: "CONTEXT: ... CONTENT: ..."                              │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 4: Estrazione Immagini da PDF                                  │
├─────────────────────────────────────────────────────────────────────┤
│ • PyMuPDF (fitz): estrae immagini bitmap embedded                   │
│ • OpenCV: layout detection per figure vettoriali                    │
│   - Render pagina a 300 DPI                                         │
│   - Canny edge detection + morphology                               │
│   - Trova contorni e croppa regioni                                 │
│ • Filtra per dimensione minima (default: 100px)                     │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 5: Analisi Immagini con Claude Opus                            │
├─────────────────────────────────────────────────────────────────────┤
│ • Ogni immagine → Claude Opus (Vision)                              │
│ • Invia: immagine base64 + testo della pagina come contesto         │
│ • Genera descrizione: tipo contenuto, componenti, valori,           │
│   connessioni, testo presente, scopo tecnico                        │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 6: Contestualizzazione Immagini                                │
├─────────────────────────────────────────────────────────────────────┤
│ • Aggiunge contesto posizionale alle descrizioni                    │
│ • "Immagine dalla pagina X del documento Y, salvata come Z"         │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 7: Calcolo Embeddings                                          │
├─────────────────────────────────────────────────────────────────────┤
│ • Sentence Transformers (all-MiniLM-L6-v2)                          │
│ • Embedding su text_for_embedding (testo contestualizzato)          │
│ • Normalizzazione L2                                                │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 8: Costruzione Index FAISS                                     │
├─────────────────────────────────────────────────────────────────────┤
│ • IndexFlatIP (Inner Product per cosine similarity)                 │
│ • Salva in: output/docs_index_multimodal_contextual.faiss           │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 9: Salvataggio Metadata                                        │
├─────────────────────────────────────────────────────────────────────┤
│ • output/metadata_multimodal_contextual.json                        │
│ • output/chunks_multimodal_contextual.jsonl                         │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 📗 2. Build con Document Intelligence (`build_index_document_intelligence.py`)

Usa **Azure AI Document Intelligence** per estrazione avanzata del layout.

```
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 1: Inizializzazione Document Intelligence                      │
├─────────────────────────────────────────────────────────────────────┤
│ • Configura client Azure Document Intelligence                      │
│ • Modello: prebuilt-layout (estrazione struttura documenti)         │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 2: Caricamento con Document Intelligence                       │
├─────────────────────────────────────────────────────────────────────┤
│ • PDF → Document Intelligence API                                   │
│ • Estrae simultaneamente:                                           │
│   - Chunk testuali con ruolo (title, sectionHeading, text)          │
│   - Tabelle con struttura (righe, colonne, celle)                   │
│   - Immagini con bounding box e caption                             │
│ • Fallback a PyPDF se DI non disponibile                            │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 3: Processing Chunk Strutturati                                │
├─────────────────────────────────────────────────────────────────────┤
│ • Usa chunk già estratti da Document Intelligence                   │
│ • Preserva ruoli: [TITLE], [SECTIONHEADING], [TEXT]                 │
│ • Mantiene riferimento a pagina di origine                          │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 4: Contextual Retrieval (Testo)                                │
├─────────────────────────────────────────────────────────────────────┤
│ • Identico a build standard                                         │
│ • Claude Opus contestualizza ogni chunk                             │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 5: Processing Tabelle (ESCLUSIVO DI)                           │
├─────────────────────────────────────────────────────────────────────┤
│ • Tabelle estratte come chunk speciali                              │
│ • Include: [TABELLA - Pagina X] + contenuto formattato              │
│ • Metadati: table_id, row_count, column_count                       │
│ • Contestualizzazione LLM della tabella                             │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 6: Gestione Immagini                                           │
├─────────────────────────────────────────────────────────────────────┤
│ • Se DI: usa immagini già estratte con caption                      │
│ • Se fallback: estrazione standard PyMuPDF + OpenCV                 │
│ • Analisi con Claude Opus (+ caption DI come contesto extra)        │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 7-9: Embeddings, FAISS, Salvataggio                            │
├─────────────────────────────────────────────────────────────────────┤
│ • Identici a build standard                                         │
│ • Chunk output includono tipo: text, image, table                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 📙 3. Build con Azure Vision (`build_index_vision.py`)

Usa **Azure AI Vision** (Image Analysis 4.0) invece di Claude per le immagini.

```
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 1-4: Identici a Build Standard                                 │
├─────────────────────────────────────────────────────────────────────┤
│ • Caricamento documenti                                             │
│ • Chunking testo                                                    │
│ • Contextual Retrieval testo (Claude)                               │
│ • Estrazione immagini (PyMuPDF + OpenCV)                            │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 5: Analisi Immagini con Azure AI Vision                        │
├─────────────────────────────────────────────────────────────────────┤
│ • Ogni immagine → Azure Vision API (Image Analysis 4.0)             │
│ • Features estratte:                                                │
│   📷 CAPTION: descrizione principale (es: "a diagram showing...")   │
│   📍 DENSE CAPTIONS: descrizioni dettagliate di regioni             │
│   🏷️ TAGS: etichette semantiche (diagram, text, table, etc.)        │
│   📝 READ (OCR): estrae tutto il testo visibile                     │
│ • Combina risultati in descrizione strutturata                      │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 6: Correlazione con Contesto Pagina                            │
├─────────────────────────────────────────────────────────────────────┤
│ • Estrae keywords dal testo della pagina                            │
│ • Cerca match tra keywords pagina e tags Azure Vision               │
│ • Aggiunge sezione "Correlazioni con la pagina"                     │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 7: Contestualizzazione Immagini                                │
├─────────────────────────────────────────────────────────────────────┤
│ • Aggiunge: "Analizzata con Azure AI Vision"                        │
│ • Contesto posizionale (pagina, documento, path)                    │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 8-10: Embeddings, FAISS, Salvataggio                           │
├─────────────────────────────────────────────────────────────────────┤
│ • Identici a build standard                                         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Confronto Build Types

| Feature | Standard | Document Intelligence | Azure Vision |
|---------|----------|----------------------|--------------|
| **Estrazione Testo** | PyPDF | Azure DI (layout avanzato) | PyPDF |
| **Analisi Immagini** | Claude Opus | Claude Opus | Azure Vision |
| **Supporto Tabelle** | ❌ No | ✅ Sì (strutturate) | ❌ No |
| **OCR Immagini** | Claude | Claude | Azure Vision Read |
| **Caption Automatici** | ❌ No | ✅ Sì (DI) | ✅ Sì (Vision) |
| **Costo Relativo** | 💰💰 | 💰💰💰 | 💰 |
| **Qualità Descrizioni** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

---

## 🔍 Ricerca (Query)

Quando fai una domanda (`rag_query.py`):

1. **Embedding query** → Sentence Transformers
2. **Ricerca FAISS** → trova top-k chunk (testo + immagini)
3. **Re-ranking immagini** → filtra per rilevanza:
   - Score ≥ 0.60 + keyword overlap ≥ 20%
   - Score ≥ 0.70 senza keyword
   - Keyword overlap ≥ 40%
4. **Build prompt** → combina 3 testi + max 5 immagini
5. **LLM Response** → Claude genera risposta
6. **Selezione immagini output** → mostra solo quelle citate o ad alto score

---

## 🚀 Utilizzo

### Build Index

```bash
# Standard (Claude per tutto)
python build_index.py --docs ./docs

# Con Document Intelligence (estrazione avanzata)
python build_index_document_intelligence.py --docs ./docs --use-document-intelligence

# Con Azure Vision (alternativa più economica per immagini)
python build_index_vision.py --docs ./docs
```

### Query Interattiva

```bash
python rag_query.py
```

---

## 📁 File Generati

Tutti i file generati vengono salvati nella cartella `output/`:

- `output/docs_index_multimodal_contextual.faiss` - Vector store con embeddings
- `output/chunks_multimodal_contextual.jsonl` - Tutti i chunk (testo + immagini) con contesto
- `output/metadata_multimodal_contextual.json` - Metadata per retrieval
- `output/extracted_images/*.png` - Immagini estratte dai PDF

---

## ⚙️ Configurazione

Tutte le configurazioni sono centralizzate in `config.py`:

### Parametri Configurabili

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `CHUNK_SIZE` | 800 | Dimensione chunk testo |
| `CHUNK_OVERLAP` | 200 | Overlap tra chunk |
| `BATCH_SIZE` | 10 | Chunk per chiamata LLM |
| `MIN_IMAGE_SIZE` | 100 | Filtra immagini piccole (px) |
| `TOP_K_TEXT` | 3 | Chunk testo da recuperare |
| `TOP_K_IMAGES` | 5 | Max immagini da recuperare |
| `OUTPUT_DIR` | "output" | Cartella file generati |

### Domain Prompt Personalizzato

Crea `domain_prompt_custom.txt` per ottimizzare il contesto nel tuo dominio:

```txt
You are an expert in [YOUR DOMAIN].
Key concepts: [technical terms, standards, procedures]

For each chunk, provide 1-2 sentence context focusing on:
- Technical category (hardware/software/procedure)
- Relationships to other sections
- Key identifiers (model numbers, versions)
```

---

## 🔧 Architettura Tecnica

**Estrazione Immagini:**
- Bitmap embedded (foto, scansioni)
- Layout detection con OpenCV (disegni tecnici vettoriali, schemi, diagrammi)

**Analisi:**
- Claude Opus 4-6 analizza ogni immagine e genera descrizione dettagliata
- Azure Vision come alternativa economica

**Storage:**
- Index FAISS unificato con embeddings di testo + descrizioni immagini
- Sentence Transformers (all-MiniLM-L6-v2) per embeddings