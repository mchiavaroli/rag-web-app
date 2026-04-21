# document_intelligence_extractor.py
"""
Modulo per l'estrazione di testo e layout da PDF usando Azure AI Document Intelligence
Integra il servizio di Document Intelligence nel sistema RAG per analisi avanzata del layout
"""
import os
import time
from typing import Dict, List, Tuple, Optional
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from PIL import Image
import io
from config import DOCUMENT_INTELLIGENCE, get_images_folder


class DocumentIntelligenceExtractor:
    """Classe per gestire l'estrazione con Azure Document Intelligence"""
    
    def __init__(self, endpoint: str = None, api_key: str = None):
        """
        Inizializza il client Document Intelligence
        
        Args:
            endpoint: Endpoint del servizio (default da config)
            api_key: Chiave API (default da config)
        """
        self.endpoint = endpoint or DOCUMENT_INTELLIGENCE['endpoint']
        self.api_key = api_key or DOCUMENT_INTELLIGENCE['api_key']
        
        # Validazione configurazione
        if '<YOUR-' in self.endpoint or '<YOUR-' in self.api_key:
            raise ValueError(
                "⚠️ Configurazione Document Intelligence non valida!\n"
                "Aggiorna endpoint e api_key in config.py"
            )
        
        self.client = DocumentAnalysisClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.api_key)
        )
        
    def extract_from_pdf(self, pdf_path: str, output_folder: str = None) -> Dict:
        """
        Estrae testo, layout e immagini da un PDF usando Document Intelligence
        
        Args:
            pdf_path: Path al file PDF
            output_folder: Cartella dove salvare le immagini estratte (default da config)
            
        Returns:
            Dizionario con:
            - 'text_chunks': Lista di chunk di testo con contesto
            - 'images': Lista di immagini estratte con metadati
            - 'tables': Lista di tabelle estratte
            - 'pages': Informazioni per pagina
        """
        # Usa la cartella di output da config se non specificata
        if output_folder is None:
            output_folder = get_images_folder()
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"📄 Analisi PDF con Document Intelligence: {pdf_path}")
        
        # Analizza documento
        # Nota: la feature "figures" richiede tier S0 (a pagamento)
        # Con tier F0 (gratuito), le immagini sono estratte con PyMuPDF
        with open(pdf_path, "rb") as f:
            poller = self.client.begin_analyze_document(
                model_id="prebuilt-layout",
                document=f
            )
        
        result = poller.result()
        
        # Estrai informazioni
        filename = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # ===== SALVA LOG OUTPUT DOCUMENT INTELLIGENCE =====
        self._save_di_log(result, filename, output_folder)
        
        # Organizza per pagine
        pages_info = self._extract_pages_info(result, filename)
        
        # Estrai testo strutturato
        text_chunks = self._extract_text_chunks(result, filename)
        
        # Estrai immagini (dalle figure rilevate)
        images = self._extract_figures(result, pdf_path, output_folder, filename)
        
        # Estrai tabelle
        tables = self._extract_tables(result, filename)
        
        return {
            'text_chunks': text_chunks,
            'images': images,
            'tables': tables,
            'pages': pages_info,
            'total_pages': len(result.pages) if result.pages else 0
        }
    
    def _save_di_log(self, result, filename: str, output_folder: str):
        """Salva il log completo dell'output di Document Intelligence"""
        import json
        
        log_path = os.path.join(output_folder, f"{filename}_document_intelligence_log.json")
        
        log_data = {
            'api_version': getattr(result, 'api_version', 'N/A'),
            'model_id': getattr(result, 'model_id', 'N/A'),
            'content_length': len(result.content) if result.content else 0,
            'content_preview': result.content[:2000] if result.content else '',
            
            # Pagine
            'pages_count': len(result.pages) if result.pages else 0,
            'pages': [],
            
            # Paragrafi
            'paragraphs_count': len(result.paragraphs) if hasattr(result, 'paragraphs') and result.paragraphs else 0,
            'paragraphs': [],
            
            # Tabelle
            'tables_count': len(result.tables) if hasattr(result, 'tables') and result.tables else 0,
            'tables': [],
            
            # Figure
            'figures_count': len(result.figures) if hasattr(result, 'figures') and result.figures else 0,
            'figures': [],
            
            # Altri attributi disponibili
            'available_attributes': [attr for attr in dir(result) if not attr.startswith('_')]
        }
        
        # Dettagli pagine
        if result.pages:
            for i, page in enumerate(result.pages):
                page_info = {
                    'page_number': i + 1,
                    'width': page.width,
                    'height': page.height,
                    'unit': page.unit,
                    'lines_count': len(page.lines) if page.lines else 0,
                    'words_count': len(page.words) if page.words else 0,
                    'selection_marks_count': len(page.selection_marks) if hasattr(page, 'selection_marks') and page.selection_marks else 0
                }
                log_data['pages'].append(page_info)
        
        # Dettagli paragrafi
        if hasattr(result, 'paragraphs') and result.paragraphs:
            for i, para in enumerate(result.paragraphs):
                para_info = {
                    'index': i,
                    'role': getattr(para, 'role', None),
                    'content_preview': para.content[:200] if para.content else '',
                    'bounding_regions': []
                }
                if para.bounding_regions:
                    for region in para.bounding_regions:
                        para_info['bounding_regions'].append({
                            'page_number': region.page_number,
                            'polygon_points': len(region.polygon) if region.polygon else 0
                        })
                log_data['paragraphs'].append(para_info)
        
        # Dettagli tabelle
        if hasattr(result, 'tables') and result.tables:
            for i, table in enumerate(result.tables):
                table_info = {
                    'index': i,
                    'row_count': table.row_count,
                    'column_count': table.column_count,
                    'cells_count': len(table.cells) if table.cells else 0,
                    'bounding_regions': []
                }
                if table.bounding_regions:
                    for region in table.bounding_regions:
                        table_info['bounding_regions'].append({
                            'page_number': region.page_number
                        })
                log_data['tables'].append(table_info)
        
        # Dettagli figure
        if hasattr(result, 'figures') and result.figures:
            for i, figure in enumerate(result.figures):
                figure_info = {
                    'index': i,
                    'id': getattr(figure, 'id', None),
                    'caption': None,
                    'bounding_regions': [],
                    'all_attributes': [attr for attr in dir(figure) if not attr.startswith('_')]
                }
                if hasattr(figure, 'caption') and figure.caption:
                    figure_info['caption'] = getattr(figure.caption, 'content', str(figure.caption))
                if figure.bounding_regions:
                    for region in figure.bounding_regions:
                        figure_info['bounding_regions'].append({
                            'page_number': region.page_number,
                            'polygon': [{'x': p.x, 'y': p.y} for p in region.polygon] if region.polygon else []
                        })
                log_data['figures'].append(figure_info)
        
        # Salva log
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"     📋 Log Document Intelligence salvato: {log_path}")
    
    def _extract_pages_info(self, result, filename: str) -> List[Dict]:
        """Estrae informazioni generali per ogni pagina"""
        pages = []
        
        for page_idx, page in enumerate(result.pages):
            page_num = page_idx + 1
            
            # Estrai tutto il testo della pagina
            page_text = []
            if page.lines:
                for line in page.lines:
                    page_text.append(line.content)
            
            pages.append({
                'page_number': page_num,
                'width': page.width,
                'height': page.height,
                'unit': page.unit,
                'text': '\n'.join(page_text),
                'line_count': len(page.lines) if page.lines else 0,
                'word_count': len(page.words) if page.words else 0,
                'source_doc': filename
            })
        
        return pages
    
    def _extract_text_chunks(self, result, filename: str) -> List[Dict]:
        """
        Estrae chunk di testo con struttura e contesto
        Mantiene informazioni su paragrafi, titoli, etc.
        """
        chunks = []
        
        # Usa i paragrafi se disponibili (miglior struttura semantica)
        paragraphs = getattr(result, 'paragraphs', None)
        if paragraphs:
            for para_idx, paragraph in enumerate(paragraphs):
                # Identifica il tipo di paragrafo
                role = getattr(paragraph, 'role', 'text')
                
                # Trova la pagina
                page_nums = []
                if paragraph.bounding_regions:
                    for region in paragraph.bounding_regions:
                        page_nums.append(region.page_number)
                page_num = page_nums[0] if page_nums else 1
                
                chunks.append({
                    'content': paragraph.content,
                    'page': page_num,
                    'role': role,  # title, sectionHeading, text, etc.
                    'source_doc': filename,
                    'chunk_id': f"{filename}_para_{para_idx}",
                    'bounding_regions': [
                        {
                            'page': region.page_number,
                            'polygon': [{'x': p.x, 'y': p.y} for p in region.polygon]
                        }
                        for region in (paragraph.bounding_regions or [])
                    ]
                })
        else:
            # Fallback: usa le pagine divise in chunk
            for page_idx, page in enumerate(result.pages):
                page_num = page_idx + 1
                
                if page.lines:
                    page_text = '\n'.join([line.content for line in page.lines])
                    
                    # Divide in chunk di ~800 caratteri
                    chunk_size = 800
                    overlap = 200
                    
                    for i in range(0, len(page_text), chunk_size - overlap):
                        chunk_text = page_text[i:i + chunk_size]
                        if chunk_text.strip():
                            chunks.append({
                                'content': chunk_text,
                                'page': page_num,
                                'role': 'text',
                                'source_doc': filename,
                                'chunk_id': f"{filename}_p{page_num}_chunk_{i}",
                                'bounding_regions': []
                            })
        
        return chunks
    
    def _extract_figures(self, result, pdf_path: str, output_folder: str, filename: str) -> List[Dict]:
        """
        Estrae figure/immagini identificate da Document Intelligence
        
        Note: Document Intelligence identifica le REGIONI delle figure ma non estrae i pixel.
        Per l'estrazione effettiva dell'immagine, manteniamo PyMuPDF.
        Se 'figures' non è disponibile, usa estrazione PyMuPDF standard.
        """
        images = []
        
        # Verifica se figures è disponibile (dipende dalla versione API)
        figures = getattr(result, 'figures', None)
        
        if not figures:
            # Fallback: usa PyMuPDF per estrazione immagini standard
            print(f"     ℹ️  'figures' non disponibile, uso estrazione PyMuPDF...")
            return self._extract_images_with_pymupdf(pdf_path, output_folder, filename)
        
        # Usa PyMuPDF per estrarre le immagini effettive dalle regioni identificate
        import fitz
        doc = fitz.open(pdf_path)
        
        for fig_idx, figure in enumerate(figures):
            # Trova la pagina della figura
            if not figure.bounding_regions:
                continue
                
            region = figure.bounding_regions[0]
            page_num = region.page_number
            page = doc[page_num - 1]
            
            # Converti le coordinate (Document Intelligence usa punti normalizzati)
            # Il formato polygon è [Point, Point, Point, Point] dove Point ha x, y
            polygon = region.polygon
            if len(polygon) < 4:
                continue
            
            # Estrai bounding box
            x_coords = [p.x for p in polygon]
            y_coords = [p.y for p in polygon]
            x0, y0 = min(x_coords), min(y_coords)
            x1, y1 = max(x_coords), max(y_coords)
            
            # Crea rettangolo per crop
            rect = fitz.Rect(x0, y0, x1, y1)
            
            # Renderizza la regione come immagine
            mat = fitz.Matrix(2, 2)  # 2x zoom per qualità
            pix = page.get_pixmap(matrix=mat, clip=rect)
            
            # Salva immagine
            img_filename = f"{filename}_p{page_num}_figure{fig_idx+1}.png"
            img_path = os.path.join(output_folder, img_filename)
            pix.save(img_path)
            
            # Estrai caption se disponibile
            caption = ""
            if hasattr(figure, 'caption') and figure.caption:
                caption = figure.caption.content if hasattr(figure.caption, 'content') else str(figure.caption)
            
            images.append({
                'page': page_num,
                'image_path': img_path,
                'size': (pix.width, pix.height),
                'source_doc': filename,
                'type': 'figure_detected',
                'caption': caption,
                'bbox': {
                    'x0': x0, 'y0': y0,
                    'x1': x1, 'y1': y1
                },
                'figure_id': getattr(figure, 'id', f"fig_{fig_idx}")
            })
        
        doc.close()
        return images
    
    def _extract_images_with_pymupdf(self, pdf_path: str, output_folder: str, filename: str, min_size: int = 100) -> List[Dict]:
        """
        Fallback: estrae immagini usando PyMuPDF quando Document Intelligence
        non fornisce l'attributo 'figures'
        """
        import fitz
        from PIL import Image
        from io import BytesIO
        
        images = []
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()
            
            for img_idx, img in enumerate(image_list):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    pil_img = Image.open(BytesIO(image_bytes))
                    width, height = pil_img.size
                    
                    if width < min_size or height < min_size:
                        continue
                    
                    img_filename = f"{filename}_p{page_num+1}_img{img_idx+1}.{image_ext}"
                    img_path = os.path.join(output_folder, img_filename)
                    
                    with open(img_path, "wb") as f:
                        f.write(image_bytes)
                    
                    images.append({
                        'page': page_num + 1,
                        'image_path': img_path,
                        'size': (width, height),
                        'source_doc': filename,
                        'type': 'bitmap',
                        'caption': '',
                        'bbox': None
                    })
                    
                except Exception as e:
                    print(f"[WARN] Errore estrazione immagine p{page_num+1} idx{img_idx}: {e}")
        
        doc.close()
        print(f"     ✓ Estratte {len(images)} immagini con PyMuPDF")
        return images

    def _extract_tables(self, result, filename: str) -> List[Dict]:
        """Estrae tabelle con struttura completa (celle, righe, colonne)"""
        tables = []
        
        # Verifica se tables è disponibile
        result_tables = getattr(result, 'tables', None)
        if not result_tables:
            return tables
        
        for table_idx, table in enumerate(result_tables):
            # Trova la pagina
            page_num = 1
            if table.bounding_regions:
                page_num = table.bounding_regions[0].page_number
            
            # Costruisci matrice della tabella
            table_data = []
            current_row = []
            last_row_idx = -1
            
            for cell in table.cells:
                # Nuova riga?
                if cell.row_index != last_row_idx:
                    if current_row:
                        table_data.append(current_row)
                    current_row = []
                    last_row_idx = cell.row_index
                
                current_row.append({
                    'content': cell.content,
                    'row': cell.row_index,
                    'col': cell.column_index,
                    'row_span': getattr(cell, 'row_span', 1),
                    'col_span': getattr(cell, 'column_span', 1),
                    'kind': getattr(cell, 'kind', 'content')  # header, content, etc.
                })
            
            # Aggiungi ultima riga
            if current_row:
                table_data.append(current_row)
            
            # Converti in formato testo per ricerca
            table_text_rows = []
            for row in table_data:
                row_text = ' | '.join([cell['content'] for cell in row])
                table_text_rows.append(row_text)
            
            table_text = '\n'.join(table_text_rows)
            
            tables.append({
                'page': page_num,
                'table_id': f"{filename}_table_{table_idx}",
                'row_count': table.row_count,
                'column_count': table.column_count,
                'data': table_data,
                'text': table_text,
                'source_doc': filename,
                'bounding_regions': [
                    {
                        'page': region.page_number,
                        'polygon': [{'x': p.x, 'y': p.y} for p in region.polygon]
                    }
                    for region in (table.bounding_regions or [])
                ]
            })
        
        return tables


def analyze_pdf_with_document_intelligence(pdf_path: str, output_folder: str = 'extracted_images') -> Dict:
    """
    Funzione helper per analizzare un PDF con Document Intelligence
    
    Args:
        pdf_path: Path al PDF
        output_folder: Cartella per salvare immagini estratte
        
    Returns:
        Dizionario con testo, immagini, tabelle estratte
    """
    extractor = DocumentIntelligenceExtractor()
    return extractor.extract_from_pdf(pdf_path, output_folder)


# ===== ESEMPIO DI UTILIZZO =====
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python document_intelligence_extractor.py <path_to_pdf>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    if not os.path.exists(pdf_path):
        print(f"❌ File non trovato: {pdf_path}")
        sys.exit(1)
    
    print("🚀 Avvio estrazione con Document Intelligence...\n")
    
    try:
        result = analyze_pdf_with_document_intelligence(pdf_path)
        
        print(f"\n✅ Estrazione completata!")
        print(f"   📄 Pagine: {result['total_pages']}")
        print(f"   📝 Chunk di testo: {len(result['text_chunks'])}")
        print(f"   🖼️  Immagini estratte: {len(result['images'])}")
        print(f"   📊 Tabelle: {len(result['tables'])}")
        
        # Mostra esempio di chunk
        if result['text_chunks']:
            print(f"\n📝 Esempio chunk:")
            chunk = result['text_chunks'][0]
            print(f"   Pagina: {chunk['page']}")
            print(f"   Tipo: {chunk['role']}")
            print(f"   Testo: {chunk['content'][:200]}...")
        
        # Mostra esempio di tabella
        if result['tables']:
            print(f"\n📊 Esempio tabella:")
            table = result['tables'][0]
            print(f"   Pagina: {table['page']}")
            print(f"   Dimensione: {table['row_count']}x{table['column_count']}")
            print(f"   Testo: {table['text'][:200]}...")
            
    except ValueError as e:
        print(f"❌ {e}")
        print("\n📝 Per configurare Document Intelligence:")
        print("   1. Crea una risorsa Document Intelligence su Azure Portal")
        print("   2. Copia endpoint e chiave")
        print("   3. Aggiorna config.py con i tuoi valori")
    except Exception as e:
        print(f"❌ Errore durante l'estrazione: {e}")
        import traceback
        traceback.print_exc()
