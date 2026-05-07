"""
Configurazione centralizzata per il progetto LLM Wiki.

ISTRUZIONI:
  1. Copia questo file come config.py
  2. Compila tutti i valori vuoti '' con le tue credenziali Azure
  3. Non committare mai config.py (aggiungilo a .gitignore)
"""

# ============================================================================
# CONFIGURAZIONE OUTPUT
# ============================================================================

OUTPUT_DIR = "output"

# ============================================================================
# PARAMETRI DI PROCESSING
# ============================================================================

# Chunking del testo
CHUNK_SIZE = 800          # Dimensione massima di ogni chunk (caratteri)
CHUNK_OVERLAP = 200       # Overlap tra chunk consecutivi (caratteri)

# Contestualizzazione LLM
BATCH_SIZE = 25           # Chunk per chiamata LLM durante la contestualizzazione
LLM_CONCURRENCY = 4       # Chiamate LLM in parallelo per la contestualizzazione

# Estrazione immagini
MIN_IMAGE_SIZE = 100      # Dimensione minima immagini da estrarre (pixel)

# Nomi file di output (relativi a OUTPUT_DIR)
CHUNKS_FILENAME = 'chunks_multimodal_contextual.jsonl'
EXTRACTED_IMAGES_FOLDER = 'extracted_images'

# ============================================================================
# CONFIGURAZIONE MODELLI LLM (Azure AI Foundry)
# ============================================================================
# Ogni entry richiede:
#   provider:          'openai' | 'anthropic' | 'mistral' | 'mistral-ocr'
#   deployment_name:   nome del deployment in Azure AI Foundry
#   endpoint:          URL endpoint della risorsa Azure
#   api_key:           chiave API
#   max_tokens:        per Anthropic e Mistral
#   max_completion_tokens: per OpenAI (al posto di max_tokens)
#   temperature:       0 = deterministico
#   api_version:       solo per provider 'openai'
#   name:              nome visualizzato nella UI

MODEL_PROVIDERS = {
    "gpt-4o": {
        'provider': 'openai',
        'deployment_name': 'gpt-4o',
        'endpoint': '',
        'api_key': '',
        'max_completion_tokens': 4096,
        'temperature': 0,
        'api_version': '2024-12-01-preview',
        'name': 'GPT-4o',
    },
    "claude-sonnet": {
        'provider': 'anthropic',
        'deployment_name': 'claude-sonnet-4-5',
        'endpoint': '',
        'api_key': '',
        'max_tokens': 4096,
        'temperature': 0,
        'name': 'Claude Sonnet',
    },
    "claude-opus": {
        'provider': 'anthropic',
        'deployment_name': 'claude-opus-4-6',
        'endpoint': '',
        'api_key': '',
        'max_tokens': 4096,
        'temperature': 0,
        'name': 'Claude Opus',
    },
    "mistral-large": {
        'provider': 'mistral',
        'deployment_name': 'Mistral-Large-3',
        'endpoint': '',
        'api_key': '',
        'max_tokens': 4096,
        'temperature': 0,
        'name': 'Mistral Large',
    },
    "mistral-document-ai": {
        'provider': 'mistral-ocr',
        'deployment_name': 'mistral-document-ai-2505',
        'endpoint': '',
        'api_key': '',
        'max_tokens': 4096,
        'temperature': 0,
        'name': 'Mistral Document AI',
    },
}

def get_model_provider(name: str):
    return MODEL_PROVIDERS.get(name)

# Modello usato per chat e ingest wiki
DEFAULT_MODEL_NAME = "claude-opus"
MODEL_PROMPT = MODEL_PROVIDERS[DEFAULT_MODEL_NAME]

# Modello usato per analisi visiva delle immagini (deve supportare vision)
DEFAULT_IMAGE_MODEL_NAME = "mistral-document-ai"
MODEL_IMAGE_ANALYSE = MODEL_PROVIDERS[DEFAULT_IMAGE_MODEL_NAME]

# ============================================================================
# CONFIGURAZIONE AZURE AI DOCUMENT INTELLIGENCE
# ============================================================================

DOCUMENT_INTELLIGENCE = {
    'endpoint': '',
    'api_key': '',
    'api_version': '2024-02-29-preview',
    'model_id': 'prebuilt-layout',
}

# ============================================================================
# CONFIGURAZIONE WIKI (LLM Wiki layer)
# ============================================================================

WIKI_DIR = "wiki"
WIKI_SCHEMA_FILE = "schema.md"
WIKI_INDEX_FILE = "index.md"
WIKI_LOG_FILE = "log.md"
WIKI_MAX_CONTEXT_PAGES = 15    # Max pagine wiki nel contesto LLM per query
WIKI_INGEST_MAX_TOKENS = 8192  # Max caratteri del documento inviati in ingest

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

import os

def get_output_path(filename):
    return os.path.join(OUTPUT_DIR, filename)

def get_chunks_path():
    return get_output_path(CHUNKS_FILENAME)

def get_images_folder():
    return get_output_path(EXTRACTED_IMAGES_FOLDER)

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(get_images_folder(), exist_ok=True)

def get_wiki_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), WIKI_DIR)

def get_wiki_schema_path():
    return os.path.join(get_wiki_dir(), WIKI_SCHEMA_FILE)

def get_wiki_index_path():
    return os.path.join(get_wiki_dir(), WIKI_INDEX_FILE)

def get_wiki_log_path():
    return os.path.join(get_wiki_dir(), WIKI_LOG_FILE)
