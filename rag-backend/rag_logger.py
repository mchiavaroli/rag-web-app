# rag_logger.py
"""
Sistema di Logging Strutturato per RAG Multimodale
Implementa tre tipi di log:
  A. Log di Indicizzazione (Build-Time)
  B. Log di Query (Run-Time)
  C. Log di Performance e Monitoring
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import time


class RAGLogger:
    """Logger centralizzato per il sistema RAG"""
    
    def __init__(self, log_dir: str = "output/logs"):
        """
        Inizializza il logger.
        
        Args:
            log_dir: Directory dove salvare i file di log
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Path file log specifici
        self.indexing_log_path = self.log_dir / "indexing_logs.jsonl"
        self.query_log_path = self.log_dir / "query_logs.jsonl"
        self.metrics_log_path = self.log_dir / "system_metrics.jsonl"
        
        # Contatori sessione corrente
        self._session_start = datetime.now()
        self._query_count = 0
        self._total_latency_ms = 0
        self._errors = []
    
    def _write_log(self, log_path: Path, log_entry: Dict) -> None:
        """Scrive una entry di log in formato JSONL (append)"""
        with open(log_path, 'a', encoding='utf-8') as f:
            json.dump(log_entry, f, ensure_ascii=False, default=str)
            f.write('\n')
    
    def _get_timestamp(self) -> str:
        """Ritorna timestamp ISO 8601"""
        return datetime.now().isoformat() + "Z"
    
    # =========================================================================
    # A. LOG DI INDICIZZAZIONE (Build-Time)
    # =========================================================================
    
    def log_indexing_start(self, documents: List[Dict]) -> Dict:
        """
        Inizia un nuovo log di indicizzazione.
        
        Args:
            documents: Lista dei documenti da indicizzare
            
        Returns:
            context dict da passare a log_indexing_complete
        """
        return {
            "start_time": time.time(),
            "start_timestamp": self._get_timestamp(),
            "documents": [
                {
                    "filename": os.path.basename(doc.get('path', doc.get('full_path', 'unknown'))),
                    "path": doc.get('full_path', doc.get('path', 'unknown'))
                }
                for doc in documents
            ]
        }
    
    def log_indexing_complete(
        self,
        context: Dict,
        document_info: Dict,
        chunks_info: Dict,
        images_info: List[Dict],
        llm_calls_info: Dict,
        index_path: str,
        success: bool = True,
        error_message: str = None
    ) -> Dict:
        """
        Completa e salva il log di indicizzazione.
        
        Args:
            context: dict ritornato da log_indexing_start
            document_info: {filename, pages, file_type}
            chunks_info: {total, text, images}
            images_info: lista di {path, page, size, type}
            llm_calls_info: {contextualization, image_analysis, total_tokens}
            index_path: path dell'index generato
            success: se l'indicizzazione è completata con successo
            error_message: eventuale messaggio di errore
        """
        processing_time_ms = int((time.time() - context["start_time"]) * 1000)
        
        log_entry = {
            "event_type": "document_indexed",
            "timestamp": self._get_timestamp(),
            "started_at": context["start_timestamp"],
            "processing_time_ms": processing_time_ms,
            "success": success,
            "document": document_info,
            "chunks": chunks_info,
            "images_extracted": images_info,
            "llm_calls": llm_calls_info,
            "output": {
                "index_path": index_path
            }
        }
        
        if error_message:
            log_entry["error"] = error_message
        
        self._write_log(self.indexing_log_path, log_entry)
        return log_entry
    
    # =========================================================================
    # B. LOG DI QUERY (Run-Time)
    # =========================================================================
    
    def log_query_start(self, query_text: str, session_id: str = None) -> Dict:
        """
        Inizia un nuovo log di query.
        
        Args:
            query_text: testo della query utente
            session_id: ID sessione (opzionale)
            
        Returns:
            context dict da passare a log_query_complete
        """
        self._query_count += 1
        
        return {
            "start_time": time.time(),
            "query_id": f"qry_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._query_count:04d}",
            "session_id": session_id or f"sess_{datetime.now().strftime('%Y%m%d')}",
            "query_text": query_text
        }
    
    def log_query_complete(
        self,
        context: Dict,
        embedding_time_ms: float,
        retrieval_info: Dict,
        response_info: Dict,
        images_shown: List[Dict] = None,
        success: bool = True,
        error_message: str = None
    ) -> Dict:
        """
        Completa e salva il log di query.
        
        Args:
            context: dict ritornato da log_query_start
            embedding_time_ms: tempo per calcolare embedding query
            retrieval_info: {search_time_ms, chunks_retrieved, chunks_text, chunks_images, top_scores}
            response_info: {llm_model, tokens_input, tokens_output, generation_time_ms, response_text}
            images_shown: lista di {name, score, keyword_overlap}
            success: se la query è completata con successo
            error_message: eventuale messaggio di errore
        """
        total_latency_ms = int((time.time() - context["start_time"]) * 1000)
        self._total_latency_ms += total_latency_ms
        
        log_entry = {
            "event_type": "query_executed",
            "timestamp": self._get_timestamp(),
            "session_id": context["session_id"],
            "query": {
                "id": context["query_id"],
                "text": context["query_text"],
                "embedding_time_ms": round(embedding_time_ms, 2)
            },
            "retrieval": retrieval_info,
            "response": response_info,  # Ora contiene response_text completo
            "images_shown": images_shown or [],
            "total_latency_ms": total_latency_ms,
            "success": success
        }
        
        if error_message:
            log_entry["error"] = error_message
            self._errors.append({
                "timestamp": self._get_timestamp(),
                "query_id": context["query_id"],
                "error": error_message
            })
        
        self._write_log(self.query_log_path, log_entry)
        return log_entry
    
    # =========================================================================
    # C. LOG DI PERFORMANCE E MONITORING
    # =========================================================================
    
    def log_system_metrics(
        self,
        index_path: str = None,
        chunks_path: str = None,
        additional_metrics: Dict = None
    ) -> Dict:
        """
        Calcola e salva metriche di sistema.
        
        Args:
            index_path: path all'index FAISS per calcolare dimensione
            chunks_path: path al file chunks per contare vettori
            additional_metrics: metriche aggiuntive custom
        """
        # Calcola metriche base
        index_size_mb = 0
        total_vectors = 0
        
        if index_path and os.path.exists(index_path):
            index_size_mb = round(os.path.getsize(index_path) / (1024 * 1024), 2)
        
        if chunks_path and os.path.exists(chunks_path):
            with open(chunks_path, 'r', encoding='utf-8') as f:
                total_vectors = sum(1 for _ in f)
        
        # Calcola latenza media
        avg_latency = 0
        if self._query_count > 0:
            avg_latency = round(self._total_latency_ms / self._query_count, 2)
        
        # Calcola cache hit rate (placeholder - da implementare con Redis)
        cache_hit_rate = 0.0
        
        # Conta errori nelle ultime 24h (semplificato: errori sessione corrente)
        errors_24h = len(self._errors)
        
        log_entry = {
            "event_type": "system_metrics",
            "timestamp": self._get_timestamp(),
            "session_started": self._session_start.isoformat(),
            "metrics": {
                "index_size_mb": index_size_mb,
                "total_vectors": total_vectors,
                "queries_this_session": self._query_count,
                "avg_query_latency_ms": avg_latency,
                "errors_this_session": errors_24h,
                "cache_hit_rate": cache_hit_rate
            }
        }
        
        if additional_metrics:
            log_entry["metrics"].update(additional_metrics)
        
        self._write_log(self.metrics_log_path, log_entry)
        return log_entry
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_recent_logs(self, log_type: str = "query", limit: int = 10) -> List[Dict]:
        """
        Legge gli ultimi N log di un tipo specifico.
        
        Args:
            log_type: "indexing", "query", o "metrics"
            limit: numero massimo di log da ritornare
        """
        path_map = {
            "indexing": self.indexing_log_path,
            "query": self.query_log_path,
            "metrics": self.metrics_log_path
        }
        
        log_path = path_map.get(log_type)
        if not log_path or not log_path.exists():
            return []
        
        logs = []
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    logs.append(json.loads(line))
        
        return logs[-limit:]  # Ultimi N
    
    def get_session_history(self, session_id: str) -> List[Dict]:
        """
        Recupera la cronologia di una sessione specifica per memoria conversazionale.
        
        Args:
            session_id: ID della sessione
            
        Returns:
            Lista di {role: "user"/"assistant", content: str} ordinata cronologicamente
        """
        if not self.query_log_path.exists():
            return []
        
        history = []
        with open(self.query_log_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    log = json.loads(line)
                    if log.get("session_id") == session_id and log.get("success", True):
                        # Aggiungi domanda utente
                        user_query = log.get("query", {}).get("text", "")
                        if user_query:
                            history.append({
                                "role": "user",
                                "content": user_query
                            })
                        # Aggiungi risposta assistente
                        response_text = log.get("response", {}).get("response_text", "")
                        if response_text:
                            history.append({
                                "role": "assistant",
                                "content": response_text
                            })
        
        return history
    
    def get_session_history_formatted(self, session_id: str, max_turns: int = 10) -> str:
        """
        Recupera la cronologia formattata per inserimento nel prompt.
        
        Args:
            session_id: ID della sessione
            max_turns: numero massimo di scambi (domanda+risposta) da includere
            
        Returns:
            Stringa formattata con la cronologia conversazionale
        """
        history = self.get_session_history(session_id)
        
        if not history:
            return ""
        
        # Limita agli ultimi N scambi (ogni scambio = 2 messaggi)
        max_messages = max_turns * 2
        history = history[-max_messages:]
        
        formatted = []
        for msg in history:
            role_label = "UTENTE" if msg["role"] == "user" else "ASSISTENTE"
            formatted.append(f"[{role_label}]: {msg['content']}")
        
        return "\n\n".join(formatted)
    
    def get_statistics(self) -> Dict:
        """Ritorna statistiche aggregate dai log"""
        query_logs = self.get_recent_logs("query", limit=1000)
        
        if not query_logs:
            return {"message": "Nessun log disponibile"}
        
        total_queries = len(query_logs)
        successful = sum(1 for l in query_logs if l.get("success", True))
        avg_latency = sum(l.get("total_latency_ms", 0) for l in query_logs) / total_queries
        
        # Calcola distribuzione score
        all_scores = []
        for log in query_logs:
            scores = log.get("retrieval", {}).get("top_scores", [])
            all_scores.extend(scores)
        
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
        
        return {
            "total_queries": total_queries,
            "successful_queries": successful,
            "success_rate": round(successful / total_queries * 100, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "avg_retrieval_score": round(avg_score, 4),
            "queries_with_images": sum(1 for l in query_logs if l.get("images_shown"))
        }
    
    def export_logs_summary(self, output_path: str = None) -> str:
        """Esporta un summary dei log in formato JSON leggibile"""
        if output_path is None:
            output_path = str(self.log_dir / "logs_summary.json")
        
        summary = {
            "generated_at": self._get_timestamp(),
            "statistics": self.get_statistics(),
            "recent_indexing": self.get_recent_logs("indexing", 5),
            "recent_queries": self.get_recent_logs("query", 10),
            "recent_metrics": self.get_recent_logs("metrics", 5)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        return output_path


# Singleton instance per uso globale
_logger_instance: Optional[RAGLogger] = None

def get_logger(log_dir: str = "output/logs") -> RAGLogger:
    """Ritorna l'istanza singleton del logger"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = RAGLogger(log_dir)
    return _logger_instance


# =========================================================================
# DEMO / TEST
# =========================================================================
if __name__ == "__main__":
    print("="*70)
    print(" RAG Logger - Test")
    print("="*70)
    
    logger = get_logger()
    
    # Test Log Indicizzazione
    print("\n1. Test Log Indicizzazione...")
    ctx = logger.log_indexing_start([{"path": "test.pdf", "full_path": "/docs/test.pdf"}])
    time.sleep(0.1)  # Simula processing
    
    log = logger.log_indexing_complete(
        context=ctx,
        document_info={"filename": "test.pdf", "pages": 10, "file_type": "pdf"},
        chunks_info={"total": 50, "text": 40, "images": 10},
        images_info=[{"path": "img1.png", "page": 1, "size": [800, 600], "type": "bitmap"}],
        llm_calls_info={"contextualization": 5, "image_analysis": 10, "total_tokens": 50000},
        index_path="output/test.faiss"
    )
    print(f"   ✓ Log salvato: {json.dumps(log, indent=2)[:200]}...")
    
    # Test Log Query
    print("\n2. Test Log Query...")
    ctx = logger.log_query_start("Come si cambia l'olio?")
    time.sleep(0.05)  # Simula elaborazione
    
    log = logger.log_query_complete(
        context=ctx,
        embedding_time_ms=15.5,
        retrieval_info={
            "search_time_ms": 8,
            "chunks_retrieved": 5,
            "chunks_text": 3,
            "chunks_images": 2,
            "top_scores": [0.89, 0.85, 0.78, 0.72, 0.68]
        },
        response_info={
            "llm_model": "claude-opus-4-6",
            "tokens_input": 2500,
            "tokens_output": 450,
            "generation_time_ms": 3200,
            "text_preview": "Per cambiare l'olio motore..."
        },
        images_shown=[{"name": "p2_bitmap1.png", "score": 0.56, "keyword_overlap": 0.65}]
    )
    print(f"   ✓ Log salvato: {json.dumps(log, indent=2)[:200]}...")
    
    # Test Metriche
    print("\n3. Test Metriche Sistema...")
    metrics = logger.log_system_metrics(
        index_path="output/docs_index_multimodal_contextual.faiss",
        chunks_path="output/chunks_multimodal_contextual.jsonl"
    )
    print(f"   ✓ Metriche: {json.dumps(metrics['metrics'], indent=2)}")
    
    # Statistiche
    print("\n4. Statistiche Aggregate...")
    stats = logger.get_statistics()
    print(f"   {json.dumps(stats, indent=2)}")
    
    print(f"\n✅ Log salvati in: {logger.log_dir}")
    print("="*70)
