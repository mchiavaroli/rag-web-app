# build_index_document_intelligence.py
"""
Versione aggiornata di build_index.py che integra Azure AI Document Intelligence
Mantiene tutte le funzionalità originali ma offre un'estrazione più avanzata del layout
"""
import os, glob, json, re, argparse, base64
from io import BytesIO
from tqdm import tqdm

# Import sentence-transformers PRIMA di fitz per evitare conflitti
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# pypdf è opzionale - serve solo come fallback quando NON usi Document Intelligence
# from pypdf import PdfReader  # Import lazy quando necessario
from PIL import Image
import fitz  # PyMuPDF per estrazione immagini - importato DOPO sentence-transformers

from config import (MODEL_PROMPT, MODEL_IMAGE_ANALYSE, CHUNK_SIZE, CHUNK_OVERLAP,
                    BATCH_SIZE, MIN_IMAGE_SIZE, EMBEDDING_MODEL,
                    get_index_path, get_metadata_path, get_chunks_path, get_images_folder,
                    ensure_output_dir)
from anthropic import AnthropicFoundry

# Import del nuovo modulo Document Intelligence
from document_intelligence_extractor import DocumentIntelligenceExtractor


# ===== IMPORTA FUNZIONI ORIGINALI =====
# Queste funzioni vengono importate dal build_index.py originale
from build_index import (
    encode_image_to_base64,
    analyze_image_with_context,
    chunk_text,
    contextualize_text_chunks_batch,
    contextualize_image_descriptions
)


def load_documents_with_document_intelligence(folder, use_doc_intelligence=False):
    """
    Carica documenti con opzione per usare Document Intelligence per i PDF
    
    Args:
        folder: Cartella contenente i documenti
        use_doc_intelligence: Se True, usa Document Intelligence per PDF
        
    Returns:
        Lista di documenti con testo estratto e informazioni aggiuntive
    """
    docs = []
    
    # Inizializza Document Intelligence se richiesto
    di_extractor = None
    if use_doc_intelligence:
        try:
            di_extractor = DocumentIntelligenceExtractor()
            print("✅ Document Intelligence configurato e pronto")
        except ValueError as e:
            print(f"⚠️  {e}")
            print("   Verrà usato il metodo di estrazione standard")
            di_extractor = None
    
    for path in sorted(glob.glob(os.path.join(folder, '*'))):
        name = os.path.basename(path)
        ext = os.path.splitext(path)[1].lower()
        
        try:
            if ext == '.txt' or ext == '.md':
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
                docs.append({
                    'path': name,
                    'text': text.strip() if text else '',
                    'full_path': path,
                    'di_result': None
                })
                
            elif ext == '.pdf':
                # Usa Document Intelligence se disponibile
                if di_extractor:
                    print(f"\n📄 Processamento {name} con Document Intelligence...")
                    di_result = di_extractor.extract_from_pdf(path)
                    
                    # Combina il testo estratto da tutti i chunk
                    text_parts = [chunk['content'] for chunk in di_result['text_chunks']]
                    text = '\n\n'.join(text_parts)
                    
                    docs.append({
                        'path': name,
                        'text': text.strip() if text else '',
                        'full_path': path,
                        'di_result': di_result  # Mantieni risultati Document Intelligence
                    })
                    
                    print(f"   ✓ Estratti: {len(di_result['text_chunks'])} chunk, "
                          f"{len(di_result['images'])} immagini, "
                          f"{len(di_result['tables'])} tabelle")
                else:
                    # Metodo standard con PyPDF (import lazy)
                    from pypdf import PdfReader
                    reader = PdfReader(path)
                    pages = [p.extract_text() or '' for p in reader.pages]
                    text = "\n".join(pages)
                    docs.append({
                        'path': name,
                        'text': text.strip() if text else '',
                        'full_path': path,
                        'di_result': None
                    })
                    
            elif ext == '.docx':
                from docx import Document
                doc = Document(path)
                paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
                text = "\n".join(paragraphs)
                docs.append({
                    'path': name,
                    'text': text.strip() if text else '',
                    'full_path': path,
                    'di_result': None
                })
                
            elif ext in ['.xlsx', '.xls']:
                # Mantieni gestione Excel originale
                if ext == '.xlsx':
                    from openpyxl import load_workbook
                    wb = load_workbook(path, data_only=True)
                    all_sheets = []
                    for sheet in wb.sheetnames:
                        ws = wb[sheet]
                        rows = []
                        for row in ws.iter_rows(values_only=True):
                            row_text = [str(cell) for cell in row if cell is not None]
                            if row_text:
                                rows.append(" | ".join(row_text))
                        if rows:
                            all_sheets.append(f"[{sheet}]\n" + "\n".join(rows))
                    text = "\n\n".join(all_sheets)
                else:  # .xls
                    import xlrd
                    wb = xlrd.open_workbook(path)
                    all_sheets = []
                    for sheet in wb.sheets():
                        rows = []
                        for r in range(sheet.nrows):
                            row = [str(sheet.cell_value(r, c)) for c in range(sheet.ncols) if sheet.cell_value(r, c)]
                            if row:
                                rows.append(" | ".join(row))
                        if rows:
                            all_sheets.append(f"[{sheet.name}]\n" + "\n".join(rows))
                    text = "\n\n".join(all_sheets)
                    
                docs.append({
                    'path': name,
                    'text': text.strip() if text else '',
                    'full_path': path,
                    'di_result': None
                })
            else:
                continue
                
        except Exception as e:
            print(f"[WARN] Errore leggendo {path}: {e}")
            
    return docs


def build_index_with_document_intelligence(
    docs, 
    embed_model_name='all-MiniLM-L6-v2',
    index_path='docs_index_multimodal_contextual.faiss',
    meta_path='metadata_multimodal_contextual.json',
    chunk_size=800, 
    overlap=200,
    analyze_images=True,
    use_text_contextualization=True,
    batch_size=10,
    domain_prompt=None,
    min_image_size=100,
    use_document_intelligence=False
):
    """
    Build index con Document Intelligence integration
    
    Differenze rispetto alla versione originale:
    - Usa chunk estrutturati da Document Intelligence (paragrafi, titoli, etc.)
    - Le immagini sono già estratte da Document Intelligence
    - Aggiunge supporto per tabelle come elementi indicizzabili
    - Mantiene retrocompatibilità con documenti processati con metodo standard
    """
    
    model = SentenceTransformer(embed_model_name)
    claude_client = AnthropicFoundry(
        api_key=MODEL_PROMPT['api_key'],
        base_url=MODEL_PROMPT['endpoint']
    )
    
    all_chunks = []
    metadata = []
    
    print("\n" + "="*70)
    print("RAG MULTIMODALE + CONTEXTUAL RETRIEVAL + DOCUMENT INTELLIGENCE")
    print("="*70)
    
    # ===== FASE 1: PROCESSING DOCUMENTI =====
    print(f"\n[1/5] Processing {len(docs)} documenti...")
    
    for doc in tqdm(docs, desc="Documenti"):
        doc_name = doc['path']
        doc_text = doc['text']
        doc_full_path = doc['full_path']
        di_result = doc.get('di_result')
        
        # ===== GESTIONE TESTO =====
        
        if use_document_intelligence and di_result:
            # Usa i chunk strutturati da Document Intelligence
            print(f"\n  📄 {doc_name}: {len(di_result['text_chunks'])} chunk strutturati (DI)")
            text_chunks = []
            
            for di_chunk in di_result['text_chunks']:
                # Aggiungi metadati di struttura al chunk original chunk
                role_prefix = ""
                role = di_chunk.get('role') or 'text'  # Default a 'text' se None
                if role != 'text':
                    role_prefix = f"[{role.upper()}] "
                
                chunk_text = role_prefix + di_chunk['content']
                text_chunks.append(chunk_text)
        else:
            # Metodo standard: chunking del testo
            text_chunks = chunk_text(doc_text, chunk_size=chunk_size, overlap=overlap)
            print(f"\n  📄 {doc_name}: {len(text_chunks)} chunk di testo")
        
        # Contestualizzazione testo
        if use_text_contextualization and text_chunks:
            print(f"     Contestualizzazione chunk testo...")
            text_chunks_contextualized = contextualize_text_chunks_batch(
                doc_text, text_chunks, doc_name, 
                batch_size=batch_size, 
                domain_prompt=domain_prompt,
                client=claude_client
            )
        else:
            text_chunks_contextualized = [f"CONTENT: {c}" for c in text_chunks]
        
        # Aggiungi chunk di testo
        for i, (original, contextualized) in enumerate(zip(text_chunks, text_chunks_contextualized)):
            # Estrai numero pagina se disponibile da DI
            page = None
            if use_document_intelligence and di_result and i < len(di_result['text_chunks']):
                page = di_result['text_chunks'][i].get('page')
            
            all_chunks.append({
                'type': 'text',
                'source': doc_name,
                'chunk_id': len(all_chunks),
                'text_original': original,
                'text_for_embedding': contextualized,
                'page': page
            })
        
        # ===== GESTIONE TABELLE (nuovo con Document Intelligence) =====
        
        if use_document_intelligence and di_result and di_result['tables']:
            print(f"     📊 {len(di_result['tables'])} tabelle trovate")
            
            for table in di_result['tables']:
                # Le tabelle vengono indicizzate come chunk di testo speciali
                table_text = f"[TABELLA - Pagina {table['page']}]\n{table['text']}"
                
                # Contestualizza la tabella
                if use_text_contextualization:
                    table_contextualized = contextualize_text_chunks_batch(
                        doc_text, [table_text], doc_name,
                        batch_size=1,
                        domain_prompt=domain_prompt,
                        client=claude_client
                    )[0]
                else:
                    table_contextualized = f"CONTENT: {table_text}"
                
                all_chunks.append({
                    'type': 'table',
                    'source': doc_name,
                    'chunk_id': len(all_chunks),
                    'text_original': table_text,
                    'text_for_embedding': table_contextualized,
                    'page': table['page'],
                    'table_id': table['table_id'],
                    'table_dimensions': f"{table['row_count']}x{table['column_count']}"
                })
        
        # ===== GESTIONE IMMAGINI =====
        
        if use_document_intelligence and di_result and di_result['images']:
            # Usa immagini estratte da Document Intelligence
            images = di_result['images']
            print(f"     🖼️  {len(images)} immagini estratte (DI)")
            
        elif doc_full_path.endswith('.pdf'):
            # Metodo standard: estrai immagini con PyMuPDF
            from build_index import extract_images_from_pdf
            print(f"     Estrazione immagini (metodo standard)...")
            images = extract_images_from_pdf(doc_full_path, output_folder=get_images_folder(), min_size=min_image_size)
            print(f"     🖼️  {len(images)} immagini estratte")
        else:
            images = []
        
        # Analizza immagini
        if analyze_images and images:
            print(f"     Analisi immagini con Claude...")
            
            # Ottieni testo pagine per contesto
            page_texts = {}
            if use_document_intelligence and di_result:
                # Usa il testo già estratto da Document Intelligence
                for page_info in di_result['pages']:
                    page_texts[page_info['page_number']] = page_info['text']
            else:
                # Estrai con PyPDF (import lazy)
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(doc_full_path)
                    for page_idx, page in enumerate(reader.pages):
                        page_texts[page_idx + 1] = page.extract_text() or ''
                except Exception as e:
                    print(f"     [WARN] Errore estrazione testo pagine: {e}")
            
            # Analizza ogni immagine
            image_descriptions = []
            for img in tqdm(images, desc="     - Analisi immagini", leave=False):
                page_text = page_texts.get(img['page'], '')
                
                # Usa caption di Document Intelligence se disponibile
                if 'caption' in img and img['caption']:
                    # Arricchisci con analisi Claude basandosi sul caption
                    description = analyze_image_with_context(
                        img['image_path'],
                        img['page'], doc_name, 
                        page_text + f"\n\nCaption documento: {img['caption']}"
                    )
                else:
                    description = analyze_image_with_context(
                        img['image_path'],
                        img['page'], doc_name, page_text
                    )
                
                image_descriptions.append({
                    'page': img['page'],
                    'image_path': img['image_path'],
                    'description': description,
                    'size': img['size']
                })
            
            # Contestualizza descrizioni immagini
            image_descriptions_contextualized = contextualize_image_descriptions(
                image_descriptions, doc_name
            )
            
            # Aggiungi chunk di immagini
            for img_data in image_descriptions_contextualized:
                all_chunks.append({
                    'type': 'image',
                    'source': doc_name,
                    'chunk_id': len(all_chunks),
                    'text_original': img_data['description'],
                    'text_for_embedding': img_data['description_contextualized'],
                    'page': img_data['page'],
                    'image_path': img_data['image_path'],
                    'image_size': img_data['size']
                })
    
    if not all_chunks:
        raise ValueError("❌ Nessun chunk generato!")
    
    # ===== FASE 2: STATISTICHE =====
    print(f"\n[2/5] Statistiche chunk...")
    text_chunks_count = sum(1 for c in all_chunks if c['type'] == 'text')
    image_chunks_count = sum(1 for c in all_chunks if c['type'] == 'image')
    table_chunks_count = sum(1 for c in all_chunks if c['type'] == 'table')
    print(f"  📄 Chunk testo: {text_chunks_count}")
    print(f"  🖼️  Chunk immagini: {image_chunks_count}")
    print(f"  📊 Chunk tabelle: {table_chunks_count}")
    print(f"  📈 Totale: {len(all_chunks)}")
    
    # ===== FASE 3: EMBEDDINGS =====
    print(f"\n[3/5] Calcolo embeddings...")
    texts_for_embedding = [c['text_for_embedding'] for c in all_chunks]
    embeddings = model.encode(texts_for_embedding, show_progress_bar=True, convert_to_numpy=True)
    
    # Normalizza
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    embeddings = embeddings / norms
    
    # ===== FASE 4: INDEX FAISS =====
    print(f"\n[4/5] Costruzione index FAISS...")
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings.astype('float32'))
    faiss.write_index(index, index_path)
    
    # ===== FASE 5: SALVATAGGIO METADATA =====
    print(f"\n[5/5] Salvataggio metadata e chunk...")
    
    for chunk in all_chunks:
        meta = {
            'type': chunk['type'],
            'source': chunk['source'],
            'chunk_id': chunk['chunk_id'],
            'text_preview': chunk['text_original'][:200]
        }
        if chunk['type'] == 'image':
            meta['page'] = chunk['page']
            meta['image_path'] = chunk['image_path']
        elif chunk['type'] == 'table':
            meta['page'] = chunk['page']
            meta['table_id'] = chunk.get('table_id')
            meta['table_dimensions'] = chunk.get('table_dimensions')
        metadata.append(meta)
    
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({'metadata': metadata}, f, ensure_ascii=False, indent=2)
    
    # Salva chunk completi
    chunks_file = get_chunks_path()
    with open(chunks_file, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            chunk_data = {
                'chunk_id': chunk['chunk_id'],
                'type': chunk['type'],
                'source': chunk['source'],
                'text_original': chunk['text_original'],
                'text_contextualized': chunk['text_for_embedding'],
                'page': chunk.get('page')
            }
            
            if chunk['type'] == 'image':
                chunk_data['image_path'] = chunk.get('image_path')
            elif chunk['type'] == 'table':
                chunk_data['table_id'] = chunk.get('table_id')
                chunk_data['table_dimensions'] = chunk.get('table_dimensions')
            
            json.dump(chunk_data, f, ensure_ascii=False)
            f.write('\n')
    
    print("\n" + "="*70)
    print("✅ COMPLETATO!")
    print("="*70)
    print(f"📁 Index FAISS: {index_path}")
    print(f"📁 Metadata: {meta_path}")
    print(f"📁 Chunks: {chunks_file}")
    print(f"📊 Totale chunk: {len(all_chunks)}")
    print(f"   - 📄 Testo (contextual): {text_chunks_count}")
    print(f"   - 🖼️  Immagini (vision+contextual): {image_chunks_count}")
    if use_document_intelligence:
        print(f"   - 📊 Tabelle (Document Intelligence): {table_chunks_count}")
    print("="*70 + "\n")
    
    return index_path, meta_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RAG Multimodale + Contextual Retrieval + Azure Document Intelligence'
    )
    parser.add_argument('--docs', type=str, default='docs', help='Cartella documenti')
    parser.add_argument('--index', type=str, default=None, help='Path index FAISS (default: da config.py)')
    parser.add_argument('--meta', type=str, default=None, help='Path metadata JSON (default: da config.py)')
    parser.add_argument('--embed_model', type=str, default=EMBEDDING_MODEL)
    parser.add_argument('--chunk_size', type=int, default=CHUNK_SIZE)
    parser.add_argument('--overlap', type=int, default=CHUNK_OVERLAP)
    parser.add_argument('--no-analyze', action='store_true', help='Estrai ma non analizzare immagini')
    parser.add_argument('--no-text-context', action='store_true', help='Disabilita contextual retrieval per testo')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Chunk per chiamata LLM')
    parser.add_argument('--domain_prompt', type=str, help='Path file prompt personalizzato')
    parser.add_argument('--min_image_size', type=int, default=MIN_IMAGE_SIZE, help='Dimensione minima immagini (px)')
    
    # NUOVO: Flag per Document Intelligence
    parser.add_argument(
        '--use-document-intelligence', 
        action='store_true',
        help='Usa Azure Document Intelligence per estrazione avanzata da PDF'
    )
    
    args = parser.parse_args()
    
    # Crea directory output
    ensure_output_dir()
    
    # Usa paths da config se non specificati
    index_path = args.index if args.index else get_index_path()
    meta_path = args.meta if args.meta else get_metadata_path()
    
    # Carica domain prompt
    domain_prompt = None
    if args.domain_prompt and os.path.exists(args.domain_prompt):
        with open(args.domain_prompt, 'r', encoding='utf-8') as f:
            domain_prompt = f.read()
        print(f"✓ Domain prompt caricato: {args.domain_prompt}")
    
    # Carica documenti
    docs = load_documents_with_document_intelligence(
        args.docs,
        use_doc_intelligence=args.use_document_intelligence
    )
    
    if not docs:
        print(f"❌ Nessun documento in {args.docs}")
        exit(1)
    
    # Build index
    build_index_with_document_intelligence(
        docs,
        embed_model_name=args.embed_model,
        index_path=index_path,
        meta_path=meta_path,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        analyze_images=not args.no_analyze,
        use_text_contextualization=not args.no_text_context,
        batch_size=args.batch_size,
        domain_prompt=domain_prompt,
        min_image_size=args.min_image_size,
        use_document_intelligence=args.use_document_intelligence
    )
