# azure_search_client.py
"""
Client Azure AI Search per RAG Multimodale.
Gestisce creazione indice, upload chunk e ricerca vettoriale.
Fallback automatico a FAISS locale se non configurato o in caso di errore.
"""
from __future__ import annotations
from config import AZURE_SEARCH


# ============================================================================
# HELPERS DI CONFIGURAZIONE
# ============================================================================

def is_configured() -> bool:
    """Restituisce True se Azure AI Search è configurato con credenziali reali."""
    endpoint = AZURE_SEARCH.get('endpoint', '')
    api_key = AZURE_SEARCH.get('api_key', '')
    return (
        bool(endpoint) and bool(api_key)
        and '<YOUR' not in endpoint
        and '<YOUR' not in api_key
    )


def _get_credential():
    from azure.core.credentials import AzureKeyCredential
    return AzureKeyCredential(AZURE_SEARCH['api_key'])


def _get_index_client():
    from azure.search.documents.indexes import SearchIndexClient
    return SearchIndexClient(
        endpoint=AZURE_SEARCH['endpoint'],
        credential=_get_credential()
    )


def _get_search_client():
    from azure.search.documents import SearchClient
    return SearchClient(
        endpoint=AZURE_SEARCH['endpoint'],
        index_name=AZURE_SEARCH['index_name'],
        credential=_get_credential()
    )


# ============================================================================
# GESTIONE INDICE
# ============================================================================

def create_or_update_index() -> None:
    """Crea o aggiorna l'indice Azure AI Search con supporto vettoriale."""
    from azure.search.documents.indexes.models import (
        SearchIndex, SearchField, SearchFieldDataType,
        SimpleField, SearchableField,
        VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile,
    )

    dims = AZURE_SEARCH.get('embedding_dimensions', 384)
    index_name = AZURE_SEARCH['index_name']

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
        SimpleField(name="chunk_id", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
        SimpleField(name="type", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="source", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SearchableField(name="text_original", type=SearchFieldDataType.String),
        SearchableField(name="text_contextualized", type=SearchFieldDataType.String),
        SimpleField(name="page", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
        SimpleField(name="image_path", type=SearchFieldDataType.String),
        SimpleField(name="table_id", type=SearchFieldDataType.String),
        SimpleField(name="table_dimensions", type=SearchFieldDataType.String),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=dims,
            vector_search_profile_name="hnsw-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw-algo")],
        profiles=[VectorSearchProfile(name="hnsw-profile", algorithm_configuration_name="hnsw-algo")],
    )

    idx = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
    _get_index_client().create_or_update_index(idx)
    print(f"✅ Indice Azure AI Search '{index_name}' creato/aggiornato")


def delete_index() -> None:
    """Elimina l'indice (utile per re-build completo)."""
    _get_index_client().delete_index(AZURE_SEARCH['index_name'])
    print(f"🗑️  Indice '{AZURE_SEARCH['index_name']}' eliminato")


# ============================================================================
# UPLOAD
# ============================================================================

def upload_chunks(all_chunks: list, embeddings) -> None:
    """
    Carica tutti i chunk con i relativi embeddings su Azure AI Search.

    Args:
        all_chunks: lista di dict chunk (come prodotta da build_index_*)
        embeddings: numpy array shape (N, dims), già normalizzati
    """
    client = _get_search_client()
    _BATCH = 1000

    documents = []
    for chunk, emb in zip(all_chunks, embeddings):
        doc = {
            "id": str(chunk['chunk_id']),
            "chunk_id": int(chunk['chunk_id']),
            "type": chunk.get('type', 'text'),
            "source": chunk.get('source', ''),
            "text_original": chunk.get('text_original', '') or '',
            "text_contextualized": chunk.get('text_for_embedding', '') or '',
            "page": int(chunk.get('page') or 0),
            "image_path": chunk.get('image_path', '') or '',
            "table_id": chunk.get('table_id', '') or '',
            "table_dimensions": chunk.get('table_dimensions', '') or '',
            "embedding": emb.tolist(),
        }
        documents.append(doc)

    total = len(documents)
    for i in range(0, total, _BATCH):
        batch = documents[i:i + _BATCH]
        client.upload_documents(batch)
        print(f"  ✓ Azure AI Search: caricati {min(i + _BATCH, total)}/{total} chunk")

    print(f"✅ Upload completato: {total} chunk su Azure AI Search")


# ============================================================================
# RICERCA
# ============================================================================

def search_chunks(query_embedding, top_k: int = 30) -> list:
    """
    Ricerca vettoriale su Azure AI Search.

    Restituisce risultati nel formato compatibile con retrieve() di rag_query.py:
    [{'chunk': {...}, 'score': float, 'type': str, 'source': str}, ...]

    I punteggi di Azure AI Search (coseno normalizzato) sono comparabili
    con quelli di FAISS IndexFlatIP su vettori normalizzati (range 0-1).
    """
    from azure.search.documents.models import VectorizedQuery

    client = _get_search_client()
    vector_query = VectorizedQuery(
        vector=query_embedding.tolist(),
        k_nearest_neighbors=top_k,
        fields="embedding",
    )

    raw_results = client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=[
            "chunk_id", "type", "source",
            "text_original", "text_contextualized",
            "page", "image_path", "table_id", "table_dimensions",
        ],
        top=top_k,
    )

    formatted = []
    for r in raw_results:
        chunk = {
            "chunk_id": r["chunk_id"],
            "type": r["type"],
            "source": r["source"],
            "text_original": r.get("text_original", ""),
            "text_contextualized": r.get("text_contextualized", ""),
            "page": r.get("page") or None,
            "image_path": r.get("image_path") or None,
        }
        formatted.append({
            "chunk": chunk,
            "score": float(r.get("@search.score", 0.0)),
            "type": r["type"],
            "source": r["source"],
        })

    return formatted
