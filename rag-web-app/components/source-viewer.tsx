'use client'

import { useState } from 'react'
import { FileText, Image as ImageIcon, ExternalLink, ZoomIn } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog'
import type { Source } from '@/lib/types'

interface SourceViewerProps {
  source: Source
}

/** Mostra un'immagine estratta come thumbnail cliccabile con lightbox */
function ImageSource({ source }: { source: Source }) {
  const [isOpen, setIsOpen] = useState(false)
  const filename = source.path.split('/').pop() || 'Immagine'
  const label = source.title
    ? `${source.title}${source.page ? ` — p. ${source.page}` : ''}`
    : filename

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <button
          className="group relative rounded-lg overflow-hidden border border-border hover:border-primary/60 transition-colors bg-muted"
          style={{ width: 96, height: 72 }}
          title={label}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={source.path}
            alt={label}
            className="w-full h-full object-cover"
            onError={e => {
              // Fallback: nasconde l'img e mostra icona
              ;(e.currentTarget as HTMLImageElement).style.display = 'none'
            }}
          />
          <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity bg-black/40">
            <ZoomIn className="h-5 w-5 text-white" />
          </div>
          {source.page && (
            <span className="absolute bottom-0 right-0 bg-black/60 text-white text-[9px] px-1 py-0.5 rounded-tl">
              p.{source.page}
            </span>
          )}
        </button>
      </DialogTrigger>

      <DialogContent className="max-w-4xl max-h-[90vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-sm">
            <ImageIcon className="h-4 w-4 shrink-0" />
            <span className="truncate">{label}</span>
          </DialogTitle>
        </DialogHeader>

        <div className="flex-1 overflow-auto min-h-0 flex items-center justify-center bg-muted rounded-lg p-4">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={source.path}
            alt={label}
            className="max-w-full max-h-[65vh] object-contain rounded"
          />
        </div>

        {source.preview && (
          <p className="text-xs text-muted-foreground mt-2 line-clamp-3 px-1">
            {source.preview}
          </p>
        )}

        <div className="flex justify-end gap-2 pt-3 border-t border-border">
          <Button variant="outline" size="sm" asChild>
            <a href={source.path} target="_blank" rel="noopener noreferrer">
              <ExternalLink className="h-3.5 w-3.5 mr-1.5" />
              Apri originale
            </a>
          </Button>
          <Button variant="secondary" size="sm" onClick={() => setIsOpen(false)}>
            Chiudi
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}

/** Mostra un chunk PDF come link cliccabile con dialog iframe */
function PdfSource({ source }: { source: Source }) {
  const [isOpen, setIsOpen] = useState(false)
  const filename = source.path.split('/').pop() || 'Documento'
  const label = source.title || filename

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <button className="flex items-start gap-2 text-xs p-2 rounded-lg bg-muted/60 hover:bg-muted transition-colors w-full text-left border border-transparent hover:border-border">
          <FileText className="h-4 w-4 text-primary shrink-0 mt-0.5" />
          <div className="flex-1 min-w-0">
            <span className="font-medium text-foreground truncate block">
              {label}
              {source.page && (
                <span className="text-muted-foreground font-normal ml-1">p. {source.page}</span>
              )}
            </span>
            {source.preview && (
              <span className="text-muted-foreground line-clamp-2 mt-0.5 leading-snug">
                {source.preview.replace(/[^\x20-\x7E\u00C0-\u024F\u0400-\u04FF\s]/g, '')}
              </span>
            )}
          </div>
          <ExternalLink className="h-3 w-3 text-muted-foreground shrink-0 mt-0.5" />
        </button>
      </DialogTrigger>

      <DialogContent className="max-w-4xl max-h-[90vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-sm">
            <FileText className="h-4 w-4 shrink-0" />
            <span className="truncate">{label}</span>
            {source.page && <span className="text-muted-foreground shrink-0">— Pagina {source.page}</span>}
          </DialogTitle>
        </DialogHeader>

        <div className="flex-1 overflow-auto min-h-0">
          <iframe
            src={source.path}
            className="w-full h-[70vh] rounded border border-border"
            title={label}
          />
        </div>

        <div className="flex justify-end gap-2 pt-3 border-t border-border">
          <Button variant="outline" size="sm" asChild>
            <a href={source.path} target="_blank" rel="noopener noreferrer">
              <ExternalLink className="h-3.5 w-3.5 mr-1.5" />
              Apri in nuova scheda
            </a>
          </Button>
          <Button variant="secondary" size="sm" onClick={() => setIsOpen(false)}>
            Chiudi
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}

export function SourceViewer({ source }: SourceViewerProps) {
  if (source.type === 'image') return <ImageSource source={source} />
  return <PdfSource source={source} />
}
