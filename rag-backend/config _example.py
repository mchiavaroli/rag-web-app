"""
Configurazione centralizzata per il progetto RAG Multimodale
Contiene sia le configurazioni Azure AI che i parametri di processing

Rinominato da config.py per chiarezza
"""

# ============================================================================
# CONFIGURAZIONE OUTPUT
# ============================================================================

# Cartella dove salvare tutti i file generati (index, metadata, chunks, immagini)
OUTPUT_DIR = "output"

# ============================================================================
# PARAMETRI DI PROCESSING
# ============================================================================

# Chunking del testo
CHUNK_SIZE = 800          # Dimensione massima di ogni chunk di testo (caratteri)
CHUNK_OVERLAP = 200       # Overlap tra chunk consecutivi (caratteri)

# Contestualizzazione LLM
BATCH_SIZE = 10           # Numero di chunk per chiamata LLM durante la contestualizzazione

# Estrazione immagini
MIN_IMAGE_SIZE = 100      # Dimensione minima immagini da estrarre (pixel)
USE_LAYOUT_DETECTION = True  # Se True, usa OpenCV per detectare figure vettoriali

# Retrieval
TOP_K_TEXT = 3            # Numero di chunk di testo da recuperare per query
TOP_K_IMAGES = 5          # Numero massimo di immagini da recuperare per query
SEARCH_MULTIPLIER = 5     # Moltiplicatore per la ricerca iniziale (top_k * questo valore)

# Soglie per selezione immagini
IMAGE_SCORE_THRESHOLD = 0.75       # Score minimo per considerare un'immagine
IMAGE_SCORE_MIN_WITH_KEYWORD = 0.65       # Score minimo per considerare un'immagine se c'è overlap keyword
IMAGE_KEYWORD_OVERLAP_MIN = 0.20   # Overlap minimo keyword per immagini con score medio
IMAGE_KEYWORD_OVERLAP_MAX = 0.30   # Overlap massimo keyword per immagini con score basso
IMAGE_HIGH_SCORE_THRESHOLD = 0.75  # Score sopra cui l'immagine è sempre inclusa
IMAGE_KEYWORD_BOOST_THRESHOLD = 0.40  # Overlap keyword sopra cui applicare boost

# Embedding model
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # Modello Sentence Transformers per embeddings

# Nomi file di output (relativi a OUTPUT_DIR)
INDEX_FILENAME = 'docs_index_multimodal_contextual.faiss'
METADATA_FILENAME = 'metadata_multimodal_contextual.json'
CHUNKS_FILENAME = 'chunks_multimodal_contextual.jsonl'
EXTRACTED_IMAGES_FOLDER = 'extracted_images'

# ============================================================================
# CONFIGURAZIONE AZURE AI FOUNDRY - MODELLI LLM
# ============================================================================

# Configurazione per rispondere alle query utente (Claude Opus)
MODEL_PROMPT = {
    'deployment_name': '',
    'endpoint': '',
    'api_key': '',
    'max_tokens': 4096,
    'temperature': 0
}

# Configurazione per analisi immagini (Claude Opus)
MODEL_IMAGE_ANALYSE = {
    'deployment_name': '',
    'endpoint': '',
    'api_key': '',
    'max_tokens': 4096,
    'temperature': 0
}

# ============================================================================
# CONFIGURAZIONE AZURE AI DOCUMENT INTELLIGENCE
# ============================================================================

DOCUMENT_INTELLIGENCE = {
    'endpoint': '',
    'api_key': '',
    'api_version': '',
    'model_id': '' 
}

# ============================================================================
# CONFIGURAZIONE AZURE AI VISION (Image Analysis)
# ============================================================================

AZURE_VISION = {
    'endpoint': '',
    'api_key': '',
    'api_version': '',
    'features': ['caption', 'denseCaptions', 'tags', 'read'],  # Funzionalità da usare
    'language': 'it',  # Lingua per caption (it, en, etc.)
    'gender_neutral_caption': True
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

import os

def get_output_path(filename):
    """Restituisce il path completo per un file nella cartella output"""
    return os.path.join(OUTPUT_DIR, filename)

def get_index_path():
    """Path completo per il file index FAISS"""
    return get_output_path(INDEX_FILENAME)

def get_metadata_path():
    """Path completo per il file metadata JSON"""
    return get_output_path(METADATA_FILENAME)

def get_chunks_path():
    """Path completo per il file chunks JSONL"""
    return get_output_path(CHUNKS_FILENAME)

def get_images_folder():
    """Path completo per la cartella immagini estratte"""
    return get_output_path(EXTRACTED_IMAGES_FOLDER)

def ensure_output_dir():
    """Crea la cartella output se non esiste"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(get_images_folder(), exist_ok=True)
