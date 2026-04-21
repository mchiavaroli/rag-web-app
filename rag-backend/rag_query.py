# rag_query_multimodal.py
"""
RAG Multimodale - Query con supporto per testo + immagini (disegni tecnici)
"""
import json
import numpy as np
import faiss
import time
from sentence_transformers import SentenceTransformer
from config import (MODEL_PROMPT, TOP_K_TEXT, TOP_K_IMAGES, MIN_K_IMAGES, SEARCH_MULTIPLIER,
                    IMAGE_SCORE_THRESHOLD, IMAGE_KEYWORD_OVERLAP_MIN,IMAGE_KEYWORD_OVERLAP_MAX,IMAGE_SCORE_MIN_WITH_KEYWORD,
                    IMAGE_HIGH_SCORE_THRESHOLD, IMAGE_KEYWORD_BOOST_THRESHOLD,
                    get_index_path, get_chunks_path)
from llm_client import call_llm_text
from rag_logger import get_logger
import os
import re


def expand_query_with_context(query: str, session_history: list, max_keywords: int = 10) -> str:
    """
    Espande la query con keyword estratte dalla cronologia conversazionale.
    Questo migliora il retrieval per query contestuali brevi (es. "dopo quanti km?")
    
    Args:
        query: query originale dell'utente
        session_history: lista di {role, content} dalla sessione
        max_keywords: numero massimo di keyword da aggiungere
        
    Returns:
        Query espansa con contesto
    """
    if not session_history:
        return query
    
    # Stopwords italiane comuni
    stopwords = {
        'il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'uno', 'una', 'di', 'da', 'a', 'in', 
        'su', 'per', 'con', 'come', 'dove', 'quando', 'chi', 'che', 'cosa', 'quale', 
        'è', 'sono', 'ha', 'hanno', 'essere', 'fare', 'questo', 'quello', 'questi',
        'del', 'della', 'dei', 'delle', 'dal', 'dalla', 'nel', 'nella', 'sul', 'sulla',
        'al', 'alla', 'ed', 'anche', 'non', 'ma', 'se', 'più', 'molto', 'può', 'ogni',
        'tra', 'fra', 'dopo', 'prima', 'poi', 'già', 'ancora', 'sempre', 'mai', 'tanto',
        'poco', 'tutto', 'tutti', 'tutte', 'altro', 'altri', 'altre', 'stesso', 'propri',
        'quale', 'quali', 'quanto', 'quanti', 'quanta', 'quante', 'quindi', 'perché',
        'però', 'dunque', 'oppure', 'ovvero', 'cioè', 'infatti', 'inoltre', 'infine',
        'the', 'and', 'or', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
        'might', 'must', 'shall', 'can', 'need', 'dare', 'ought', 'used', 'to', 'of',
        'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'under', 'again', 'further',
        'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
        'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very', 'just', 'also'
    }
    
    # Estrai keyword dagli ultimi messaggi (priorità agli ultimi scambi)
    keyword_freq = {}
    
    # Considera solo gli ultimi 4 messaggi (2 scambi)
    recent_history = session_history[-4:] if len(session_history) > 4 else session_history
    
    for msg in recent_history:
        content = msg.get('content', '')
        # Pulisci il testo: rimuovi markdown, punteggiatura
        content = re.sub(r'[#*_\-|`>]', ' ', content)
        content = re.sub(r'[^\w\s]', ' ', content)
        
        words = content.lower().split()
        for word in words:
            # Filtra: no stopwords, lunghezza minima 4, no numeri puri
            if (word not in stopwords and 
                len(word) >= 4 and 
                not word.isdigit() and
                not word.startswith('http')):
                keyword_freq[word] = keyword_freq.get(word, 0) + 1
    
    # Ordina per frequenza e prendi le top keyword
    sorted_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)
    top_keywords = [kw for kw, freq in sorted_keywords[:max_keywords]]
    
    # Costruisci query espansa
    if top_keywords:
        # Aggiungi keyword come contesto alla query
        context_str = ' '.join(top_keywords)
        expanded = f"{query} (contesto: {context_str})"
        return expanded
    
    return query


def load_index(index_path=None, 
               chunks_path=None, 
               embed_model='all-MiniLM-L6-v2'):
    """Carica l'index FAISS multimodale + contextual e i chunks"""
    # Usa paths da config se non specificati
    if index_path is None:
        index_path = get_index_path()
    if chunks_path is None:
        chunks_path = get_chunks_path()
    
    index = faiss.read_index(index_path)
    chunks = {}
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            chunks[obj['chunk_id']] = obj
    model = SentenceTransformer(embed_model)
    return index, chunks, model


def retrieve(query, index, chunks, model, top_k=None, expanded_query=None):
    """Recupera i chunk più rilevanti (testo + immagini) con re-ranking bilanciato
    
    Args:
        query: query originale per keyword matching
        expanded_query: query espansa per embedding (se None, usa query)
    """
    if top_k is None:
        top_k = TOP_K_TEXT + TOP_K_IMAGES
    
    # Usa expanded_query per l'embedding se disponibile
    search_query = expanded_query if expanded_query else query
    
    q_emb = model.encode([search_query], convert_to_numpy=True)
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-10)
    
    # Recupera molti più risultati per avere scelta tra testi e immagini
    search_k = min(top_k * SEARCH_MULTIPLIER, len(chunks))
    D, I = index.search(q_emb.astype('float32'), search_k)
    
    results = []
    for idx, score in zip(I[0], D[0]):
        if idx < 0 or idx not in chunks:
            continue
        chunk = chunks[idx]
        results.append({
            'chunk': chunk,
            'score': float(score),
            'type': chunk['type'],
            'source': chunk['source']
        })
    
    # Separa testo e immagini
    text_results = [r for r in results if r['type'] == 'text']
    image_results = [r for r in results if r['type'] == 'image']
    
    # Re-ranking per immagini - usa soglie da config
    if image_results:
        query_lower = query.lower()
        stopwords = {'il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'uno', 'una', 'di', 'da', 'a', 'in', 'su', 'per', 'con', 'come', 'dove', 'quando', 'chi', 'che', 'cosa', 'quale', 'è', 'sono', 'ha', 'hanno'}
        query_words = set(word for word in query_lower.split() if word not in stopwords and len(word) > 2)
        
        filtered_images = []
        for img_result in image_results:
            desc = img_result['chunk'].get('text_original', '').lower()
            desc_words = set(word for word in desc.split() if len(word) > 2)
            
            keyword_overlap = len(query_words & desc_words) / max(len(query_words), 1)
            base_score = img_result['score']
            img_result['keyword_overlap'] = keyword_overlap  # Salva per debug
            
            # Usa soglie da config.py
            if base_score >= IMAGE_SCORE_THRESHOLD and keyword_overlap >= IMAGE_KEYWORD_OVERLAP_MIN:
                filtered_images.append(img_result)
            elif base_score >= IMAGE_SCORE_MIN_WITH_KEYWORD and keyword_overlap >= IMAGE_KEYWORD_OVERLAP_MAX:  # Keyword compensano score medio
                filtered_images.append(img_result)
            elif base_score >= IMAGE_HIGH_SCORE_THRESHOLD:  # Score alto anche senza keyword
                filtered_images.append(img_result)
            elif keyword_overlap >= IMAGE_KEYWORD_BOOST_THRESHOLD:  # Molte keyword in comune
                filtered_images.append(img_result)
        
        image_results = filtered_images
    
    image_results.sort(key=lambda x: x['score'], reverse=True)
    
    # Combina: TOP_K_TEXT testi + TOP_K_IMAGES immagini
    final_results = []
    final_results.extend(text_results[:TOP_K_TEXT])
    
    # Aggiungi immagini filtrate (max TOP_K_IMAGES)
    if image_results:
        final_results.extend(image_results[:TOP_K_IMAGES])
    
    # Se MIN_K_IMAGES > 0 e non abbiamo abbastanza immagini, prendi le migliori non filtrate
    current_images = len([r for r in final_results if r['type'] == 'image'])
    if MIN_K_IMAGES > 0 and current_images < MIN_K_IMAGES:
        all_images = sorted([r for r in results if r['type'] == 'image'], key=lambda x: x['score'], reverse=True)
        for img in all_images:
            if img not in final_results:
                final_results.append(img)
                current_images += 1
                if current_images >= MIN_K_IMAGES:
                    break
    
    # Riordina per score
    final_results.sort(key=lambda x: x['score'], reverse=True)
    
    return final_results[:top_k]


def build_multimodal_prompt(query, retrieved_results, conversation_history: str = None):
    """Costruisce il prompt includendo info su testo, immagini e cronologia conversazione"""
    text_chunks = []
    image_refs = []
    
    for i, result in enumerate(retrieved_results, 1):
        chunk = result['chunk']
        score = result['score']
        
        # Usa text_original invece di text (compatibile con chunk contextual)
        chunk_text = chunk.get('text_original', chunk.get('text', ''))
        
        if chunk['type'] == 'text':
            text_chunks.append(f"[DOCUMENTO {i} - {chunk['source']} - Score: {score:.3f}]\n{chunk_text}")
        elif chunk['type'] == 'image':
            img_path = chunk.get('image_path', '')
            image_refs.append(f"[IMMAGINE {i} - {chunk['source']} pag.{chunk.get('page', '?')} - Score: {score:.3f}]")
            text_chunks.append(f"[IMMAGINE {i} - {chunk['source']} pag.{chunk.get('page', '?')}]\n{chunk_text}")
    
    context = "\n\n---\n\n".join(text_chunks)
    
    # Costruisci sezione cronologia conversazionale
    history_section = ""
    if conversation_history:
        history_section = f"""
=== CRONOLOGIA CONVERSAZIONE PRECEDENTE ===
{conversation_history}
=== FINE CRONOLOGIA ===

"""
    
    prompt = f"""
Sei un assistente esperto nella ricerca di informazioni tecniche.
{history_section}Hai a disposizione i seguenti documenti (sia testo che descrizioni di DISEGNI TECNICI):

{context}

Istruzioni:
1. Usa PRINCIPALMENTE le informazioni presenti nei documenti forniti.
2. Se trovi informazioni da IMMAGINI/DISEGNI, specifica che provengono da un disegno tecnico.
3. Se la query riguarda schemi, diagrammi, planimetrie o grafici, dai priorità ai chunk marcati come [IMMAGINE].
4. Cerca corrispondenze sia esatte che parziali per codici, numeri, nomi.
5. Se non trovi informazioni rilevanti, dillo chiaramente.
6. Se c'è cronologia conversazionale, considera il contesto delle domande precedenti per dare risposte coerenti.

[RUOLO E CONTESTO] Sei un Assistente Tecnico Esperto specializzato nelle procedure di assemblaggio automotive. Il tuo compito è supportare gli operatori e i tecnici fornendo istruzioni di montaggio precise, basate ESCLUSIVAMENTE sulla documentazione tecnica e sulle Work Instructions fornite nel contesto.
[MATERIALE DI RIFERIMENTO] Le Work Instructions fornite sono composte da testo descrittivo e disegni 3D CAD. I disegni tecnici sono una parte essenziale dell'istruzione: spesso contengono informazioni vitali (come direzioni di rotazione, pinout, fori di riferimento) che NON sono ripetute nel testo.
[ISTRUZIONI OPERATIVE] Per rispondere alle domande degli utenti, devi seguire rigorosamente queste regole:
Analisi Multimodale Prioritaria: Prima di rispondere, analizza attentamente sia il testo che le immagini fornite. Cerca attivamente indicatori visivi come frecce, colori, evidenziazioni o legende all'interno dei disegni CAD.
Riferimenti Precisi: Quando descrivi un'operazione, includi sempre il Part Number (P/N), il numero dell'operazione (Operazione ID) e gli eventuali parametri tecnici (es. coppie di serraggio) associati, se presenti nel contesto.
Tolleranza Zero per le Allucinazioni: La tua priorità assoluta è l'accuratezza. NON inventare mai codici, procedure, direzioni o componenti. Se l'informazione richiesta non è deducibile né dal testo né dalle immagini fornite, devi rispondere ESCLUSIVAMENTE: "Mi dispiace, ma l'informazione richiesta non è presente nella documentazione fornita."
Stile di Risposta: Sii diretto, conciso e professionale. Usa elenchi puntati o numerati per descrivere passaggi sequenziali o liste di componenti. Non aggiungere convenevoli non necessari.
Cita immagigni di disegni tecnici presenti all'interno della documentazione.

Domanda: {query}

Risposta:
"""
    
    return prompt, image_refs


def ask_llm(query, index, chunks, model, show_sources=True, enable_logging=True, session_id=None, model_config=None):
    """Esegue la query RAG multimodale completa con memoria conversazionale. model_config permette override del modello LLM."""
    
    # === LOGGING: Inizializza ===
    logger = get_logger() if enable_logging else None
    log_context = logger.log_query_start(query, session_id=session_id) if logger else None
    embedding_time_ms = 0
    
    # === QUERY EXPANSION: Espandi query con contesto conversazionale ===
    expanded_query = query
    session_history = []
    if logger and session_id:
        session_history = logger.get_session_history(session_id)
        if session_history:
            expanded_query = expand_query_with_context(query, session_history)
            if expanded_query != query:
                print(f"🔍 Query espansa: \"{expanded_query[:80]}...\"" if len(expanded_query) > 80 else f"🔍 Query espansa: \"{expanded_query}\"")
    
    retrieval_start = time.time()
    
    # Usa la query espansa per il retrieval (migliora match immagini)
    retrieved = retrieve(query, index, chunks, model, top_k=5, expanded_query=expanded_query)
    retrieval_time_ms = (time.time() - retrieval_start) * 1000
    
    if show_sources:
        print("\n" + "="*70)
        print("📚 CHUNK RECUPERATI:")
        print("="*70)
        for i, result in enumerate(retrieved, 1):
            chunk = result['chunk']
            score = result['score']
            chunk_type = "📄 TESTO" if chunk['type'] == 'text' else "🖼️  IMMAGINE"
            kw_info = f" | Keyword: {result.get('keyword_overlap', 0)*100:.0f}%" if chunk['type'] == 'image' else ""
            print(f"\n{i}. {chunk_type} | {chunk['source']} | Score: {score*100:.0f}%{kw_info}")
            
            if chunk['type'] == 'image':
                print(f"   📍 Pagina: {chunk.get('page', '?')}")
                print(f"   🗂️  File: {chunk.get('image_path', 'N/A')}")
            
            # Usa text_original invece di text
            chunk_text = chunk.get('text_original', chunk.get('text', ''))
            preview = chunk_text[:150].replace('\n', ' ')
            print(f"   📝 Preview: {preview}...")
        print("="*70 + "\n")
    
    # === MEMORIA CONVERSAZIONALE: Formatta cronologia per il prompt ===
    conversation_history = None
    if session_history:
        conversation_history = logger.get_session_history_formatted(session_id, max_turns=5)
        if conversation_history:
            print(f"💬 Cronologia: {len(session_history)//2} scambi precedenti")
    
    prompt, image_refs = build_multimodal_prompt(query, retrieved, conversation_history)
    
    print("⏳ Interrogazione LLM in corso...")
    
    try:
        llm_start = time.time()
        config = model_config if model_config is not None else MODEL_PROMPT
        response_text, llm_usage = call_llm_text(
            config,
            system_prompt="Sei un assistente tecnico esperto, preciso e sintetico.",
            user_prompt=prompt,
        )
        llm_time_ms = (time.time() - llm_start) * 1000
        
        # === LOGGING: Salva log query ===
        if logger and log_context:
            # Calcola info retrieval
            chunks_text = sum(1 for r in retrieved if r['chunk']['type'] == 'text')
            chunks_images = sum(1 for r in retrieved if r['chunk']['type'] == 'image')
            top_scores = [r['score'] for r in retrieved[:5]]
            
            # Info immagini mostrate (verranno aggiornate nel main)
            images_shown = [
                {
                    "name": os.path.basename(r['chunk'].get('image_path', 'unknown')),
                    "score": round(r['score'], 3),
                    "keyword_overlap": round(r.get('keyword_overlap', 0), 3)
                }
                for r in retrieved if r['chunk']['type'] == 'image'
            ]
            
            logger.log_query_complete(
                context=log_context,
                embedding_time_ms=embedding_time_ms,
                retrieval_info={
                    "search_time_ms": round(retrieval_time_ms, 2),
                    "chunks_retrieved": len(retrieved),
                    "chunks_text": chunks_text,
                    "chunks_images": chunks_images,
                    "top_scores": [round(s, 4) for s in top_scores]
                },
                response_info={
                    "llm_model": config['deployment_name'],
                    "tokens_input": llm_usage.get('input_tokens'),
                    "tokens_output": llm_usage.get('output_tokens'),
                    "generation_time_ms": round(llm_time_ms, 2),
                    "response_text": response_text  # Output completo per memoria conversazionale
                },
                images_shown=images_shown
            )
        
        return response_text, retrieved, images_shown
    except Exception as e:
        # === LOGGING: Log errore ===
        images_shown = []
        if logger and log_context:
            logger.log_query_complete(
                context=log_context,
                embedding_time_ms=embedding_time_ms,
                retrieval_info={"search_time_ms": round(retrieval_time_ms, 2), "chunks_retrieved": len(retrieved)},
                response_info={"llm_model": config['deployment_name'], "error": str(e)},
                success=False,
                error_message=str(e)
            )
        print(f"\n❌ Errore nella chiamata LLM: {e}")
        raise


# ---------------------
# MAIN - Interfaccia Interattiva
# ---------------------
if __name__ == "__main__":
    print("="*70)
    print(" RAG MULTIMODALE - Testo + Immagini (Disegni Tecnici)")
    print("="*70)
    print("\nCaricamento index e modello...")
    
    try:
        index, chunks, model = load_index()
        print(f"✓ Index caricato: {index.ntotal} vettori")
        print(f"✓ Chunks caricati: {len(chunks)}")
        
        # Conta tipi
        text_count = sum(1 for c in chunks.values() if c['type'] == 'text')
        img_count = sum(1 for c in chunks.values() if c['type'] == 'image')
        
        print(f"  - 📄 Chunk testo: {text_count}")
        print(f"  - 🖼️  Chunk immagini: {img_count}")
        
        # === NUOVA SESSIONE: Genera ID univoco per questa esecuzione ===
        from datetime import datetime
        session_id = f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n🆔 Sessione: {session_id}")
        print(f"💡 Puoi fare domande su testi E disegni tecnici!")
        print(f"💡 La conversazione sarà ricordata durante questa sessione.")
        print(f"💡 Comandi speciali:")
        print(f"   - 'exit' o 'quit': esci")
        print(f"   - 'stats': mostra statistiche log")
        print(f"   - 'logs': mostra ultimi log query")
        print(f"   - 'history': mostra cronologia sessione corrente")
        print(f"   - 'metrics': salva metriche di sistema\n")
        
        # Logger per comandi speciali
        logger = get_logger()
        
        while True:
            print("-"*70)
            query = input("🔍 Domanda: ").strip()
            
            if not query:
                continue
                
            if query.lower() in ['exit', 'quit', 'esci', 'q']:
                # Salva metriche finali prima di uscire
                logger.log_system_metrics(
                    index_path=get_index_path(),
                    chunks_path=get_chunks_path()
                )
                print("\n📊 Metriche sessione salvate.")
                print("👋 Arrivederci!")
                break
            
            # Comandi speciali per log
            if query.lower() == 'stats':
                stats = logger.get_statistics()
                print("\n📊 STATISTICHE LOG:")
                print(json.dumps(stats, indent=2, ensure_ascii=False))
                continue
            
            if query.lower() == 'logs':
                recent = logger.get_recent_logs("query", limit=5)
                print(f"\n📋 ULTIMI {len(recent)} LOG QUERY:")
                for log in recent:
                    q_text = log.get('query', {}).get('text', 'N/A')[:50]
                    latency = log.get('total_latency_ms', 0)
                    success = "✅" if log.get('success', True) else "❌"
                    print(f"  {success} \"{q_text}...\" - {latency}ms")
                continue
            
            if query.lower() == 'metrics':
                metrics = logger.log_system_metrics(
                    index_path=get_index_path(),
                    chunks_path=get_chunks_path()
                )
                print("\n📊 METRICHE SISTEMA SALVATE:")
                print(json.dumps(metrics['metrics'], indent=2))
                continue
            
            if query.lower() == 'history':
                history = logger.get_session_history_formatted(session_id)
                if history:
                    print(f"\n📜 CRONOLOGIA SESSIONE {session_id}:")
                    print("-"*50)
                    print(history)
                    print("-"*50)
                else:
                    print("\n📜 Nessuna cronologia per questa sessione.")
                continue
            
            try:
                print("\n⏳ Ricerca multimodale in corso...")
                answer, retrieved = ask_llm(query, index, chunks, model, show_sources=True, session_id=session_id)
                
                print("="*70)
                print("📝 RISPOSTA:")
                print("="*70)
                print(answer)
                print("="*70 + "\n")
                
                # Mostra immagini rilevanti con logica più flessibile
                answer_lower = answer.lower()
                relevant_images = []
                
                for idx, r in enumerate(retrieved):
                    if r['chunk']['type'] == 'image' and r['chunk'].get('image_path'):
                        img_path = r['chunk']['image_path']
                        img_score = r['score']
                        
                        # Estrai info immagine
                        page = r['chunk'].get('page', '')
                        source = r['chunk'].get('source', '')
                        desc = r['chunk'].get('text_original', '').lower()
                        
                        # Considera rilevante se:
                        # 1. È tra i top 3 risultati E ha score decente (>0.65)
                        # 2. Menzionata nella risposta (riferimento a pagina/fonte/immagine)
                        # 3. Score molto alto (>0.75)
                        # 4. Parole chiave della descrizione presenti nella risposta
                        
                        is_top_result = idx < 3 and img_score >= 0.65
                        
                        is_mentioned = (f"pagina {page}" in answer_lower or 
                                       f"pag. {page}" in answer_lower or
                                       f"pag {page}" in answer_lower or
                                       "immagine" in answer_lower or
                                       "figura" in answer_lower or
                                       "disegno" in answer_lower or
                                       source.lower().replace('.pdf', '') in answer_lower)
                        
                        high_score = img_score >= 0.75
                        
                        # Check se parole chiave dell'immagine sono nella risposta
                        desc_words = set(word for word in desc.split() if len(word) > 4)
                        answer_words = set(word for word in answer_lower.split() if len(word) > 4)
                        keyword_in_answer = len(desc_words & answer_words) >= 2
                        
                        if is_top_result or is_mentioned or high_score or (keyword_in_answer and img_score >= 0.70):
                            relevant_images.append(img_path)
                
                if relevant_images:
                    print("🖼️  IMMAGINI CORRELATE:")
                    for i, path in enumerate(relevant_images, 1):
                        if os.path.exists(path):
                            print(f"   {i}. {path}")
                    print()
                
            except Exception as e:
                print(f"\n❌ Errore: {e}\n")
    
    except FileNotFoundError as e:
        print("\n❌ ERRORE: Index multimodale non trovato!")
        print("Devi prima eseguire:")
        print("  python build_index_multimodal.py --docs ./docs")
        print()
