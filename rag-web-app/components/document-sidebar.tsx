'use client'

import { useState, useCallback } from 'react'
import useSWR, { mutate } from 'swr'
import { FileText, Upload, Trash2, File, Loader2, RefreshCw, CheckCircle2, AlertCircle, Clock } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { LogViewer } from '@/components/log-viewer'
import { IndexingLogViewer } from '@/components/indexing-log-viewer'
import { Card } from '@/components/ui/card'
import type { Document, IndexStatus } from '@/lib/types'

const fetcher = (url: string) => fetch(url).then(res => res.json())

/** Polling interval in ms durante l'indicizzazione */
const POLLING_INTERVAL_BUILDING = 3000
const POLLING_INTERVAL_IDLE = 0

function IndexStatusBadge({ status }: { status: IndexStatus }) {
  if (status.status === 'building') {
    return (
      <div className="flex items-center gap-2 rounded-md bg-amber-50 dark:bg-amber-950 border border-amber-200 dark:border-amber-800 px-3 py-2 text-xs">
        <Loader2 className="h-3.5 w-3.5 animate-spin text-amber-600 shrink-0" />
        <span className="text-amber-700 dark:text-amber-300 font-medium">Indicizzazione in corso...</span>
      </div>
    )
  }
  if (status.status === 'ready') {
    return (
      <div className="flex items-center gap-2 rounded-md bg-green-50 dark:bg-green-950 border border-green-200 dark:border-green-800 px-3 py-2 text-xs">
        <CheckCircle2 className="h-3.5 w-3.5 text-green-600 shrink-0" />
        <span className="text-green-700 dark:text-green-300 font-medium">
          Indice pronto — {status.total_chunks} chunk ({status.text_chunks} testo, {status.image_chunks} img)
        </span>
      </div>
    )
  }
  if (status.status === 'error') {
    return (
      <div className="flex items-center gap-2 rounded-md bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800 px-3 py-2 text-xs">
        <AlertCircle className="h-3.5 w-3.5 text-red-600 shrink-0" />
        <span className="text-red-700 dark:text-red-300 font-medium truncate" title={status.message}>{status.message}</span>
      </div>
    )
  }
  return (
    <div className="flex items-center gap-2 rounded-md bg-muted border border-border px-3 py-2 text-xs">
      <Clock className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
      <span className="text-muted-foreground">{status.message || 'Nessun documento'}</span>
    </div>
  )
}

export function DocumentSidebar() {
  const [isDragging, setIsDragging] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [isReindexing, setIsReindexing] = useState(false)

  const { data: indexData } = useSWR<IndexStatus>(
    '/api/index',
    fetcher,
    {
      refreshInterval: (data) =>
        data?.status === 'building' ? POLLING_INTERVAL_BUILDING : POLLING_INTERVAL_IDLE,
    }
  )

  const { data, error, isLoading } = useSWR<{ documents: Document[] }>(
    '/api/documents',
    fetcher,
    {
      refreshInterval: indexData?.status === 'building' ? POLLING_INTERVAL_BUILDING : POLLING_INTERVAL_IDLE,
    }
  )

  const documents = data?.documents || []

  const handleUpload = async (files: FileList | File[]) => {
    const pdfFiles = Array.from(files).filter(
      f => f.type === 'application/pdf'
    )

    if (pdfFiles.length === 0) {
      alert('Seleziona solo file PDF')
      return
    }

    setIsUploading(true)

    const formData = new FormData()
    pdfFiles.forEach(file => formData.append('files', file))

    try {
      const response = await fetch('/api/documents/upload', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const err = await response.json().catch(() => ({}))
        throw new Error(err.error || 'Upload fallito')
      }

      mutate('/api/documents')
      mutate('/api/index')
    } catch (err) {
      console.error('Upload error:', err)
      alert(err instanceof Error ? err.message : 'Errore durante il caricamento')
    } finally {
      setIsUploading(false)
    }
  }

  const handleDelete = async (id: string) => {
    if (!confirm('Sei sicuro di voler eliminare questo documento?')) return

    try {
      const response = await fetch(`/api/documents?id=${encodeURIComponent(id)}`, {
        method: 'DELETE',
      })

      if (!response.ok) throw new Error('Eliminazione fallita')

      mutate('/api/documents')
      mutate('/api/index')
    } catch (err) {
      console.error('Delete error:', err)
      alert('Errore durante l\'eliminazione')
    }
  }

  const handleReindex = async () => {
    setIsReindexing(true)
    try {
      const response = await fetch('/api/index', { method: 'POST' })
      const data = await response.json()
      if (!data.success) alert(data.message || 'Impossibile avviare l\'indicizzazione')
      mutate('/api/index')
    } catch {
      alert('Errore durante l\'avvio dell\'indicizzazione')
    } finally {
      setIsReindexing(false)
    }
  }

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    if (e.dataTransfer.files) handleUpload(e.dataTransfer.files)
  }, [])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }


  return (
    <div className="w-80 border-r border-border bg-card flex flex-col h-full">
      <div className="p-4 border-b border-border">
        <h2 className="font-semibold text-lg flex items-center gap-2 text-card-foreground">
          <FileText className="h-5 w-5" />
          Documenti
        </h2>
      </div>

      {/* Indexing Log Viewer Button */}
      <div className="px-4 pt-3 pb-1 flex flex-col gap-2">
        <IndexingLogViewer />
        <LogViewer />
      </div>

      {/* Upload Area */}
      <div className="p-4 border-b border-border">
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          className={`
            border-2 border-dashed rounded-lg p-6 text-center transition-colors
            ${isDragging
              ? 'border-primary bg-primary/5'
              : 'border-muted-foreground/25 hover:border-primary/50'
            }
          `}
        >
          {isUploading ? (
            <div className="flex flex-col items-center gap-2">
              <Loader2 className="h-8 w-8 animate-spin text-primary" />
              <span className="text-sm text-muted-foreground">
                Caricamento in corso...
              </span>
            </div>
          ) : (
            <>
              <Upload className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
              <p className="text-sm text-muted-foreground mb-2">
                Trascina i PDF qui
              </p>
              <label>
                <input
                  type="file"
                  multiple
                  accept=".pdf"
                  className="hidden"
                  onChange={e => e.target.files && handleUpload(e.target.files)}
                />
                <Button variant="outline" size="sm" asChild>
                  <span className="cursor-pointer">Sfoglia</span>
                </Button>
              </label>
            </>
          )}
        </div>
      </div>

      {/* Index Status */}
      <div className="px-4 py-3 border-b border-border space-y-2">
        {indexData ? (
          <IndexStatusBadge status={indexData} />
        ) : (
          <div className="flex items-center gap-2 rounded-md bg-muted border border-border px-3 py-2 text-xs">
            <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground shrink-0" />
            <span className="text-muted-foreground">Connessione al backend...</span>
          </div>
        )}
        {indexData && indexData.status !== 'building' && documents.length > 0 && (
          <Button
            variant="outline"
            size="sm"
            className="w-full gap-2 text-xs"
            onClick={handleReindex}
            disabled={isReindexing}
          >
            {isReindexing
              ? <Loader2 className="h-3.5 w-3.5 animate-spin" />
              : <RefreshCw className="h-3.5 w-3.5" />
            }
            Re-indicizza tutto
          </Button>
        )}
      </div>

      {/* Documents List */}
      <div className="flex-1 overflow-y-auto">
        <div className="p-3 space-y-2">
          {isLoading ? (
            <div className="flex justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          ) : error ? (
            <p className="text-sm text-destructive text-center py-4">
              Backend non raggiungibile
            </p>
          ) : documents.length === 0 ? (
            <p className="text-sm text-muted-foreground text-center py-8">
              Nessun documento caricato
            </p>
          ) : (
            documents.map(doc => (
              <Card
                key={doc.id}
                className="p-3 group hover:bg-accent/50 transition-colors overflow-hidden"
              >
                <div className="flex items-center gap-2 min-w-0">
                  <File className="h-7 w-7 text-primary shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-xs truncate text-card-foreground" title={doc.name}>
                      {doc.name}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {formatFileSize(doc.size)} •{' '}
                      {new Date(doc.uploadedAt).toLocaleDateString('it-IT')}
                    </p>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity"
                    onClick={() => handleDelete(doc.id)}
                  >
                    <Trash2 className="h-3.5 w-3.5 text-destructive" />
                  </Button>
                </div>
              </Card>
            ))
          )}
        </div>
      </div>

      {/* Stats */}
      {documents.length > 0 && (
        <div className="p-4 border-t border-border bg-muted/30">
          <p className="text-xs text-muted-foreground text-center">
            {documents.length} documento{documents.length !== 1 ? 'i' : ''} caricato{documents.length !== 1 ? 'i' : ''}
          </p>
        </div>
      )}
    </div>
  )
}

