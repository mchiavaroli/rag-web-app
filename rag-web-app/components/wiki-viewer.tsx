'use client'

import { useState, useCallback } from 'react'
import useSWR, { mutate } from 'swr'
import {
  BookOpen, FileText, Upload, Search, AlertTriangle,
  RefreshCw, Trash2, ChevronRight, Layers, Wrench, Box, BookMarked, ImageIcon, ZoomIn, GraduationCap
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { Textarea } from '@/components/ui/textarea'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { WikiStatus, WikiPage } from '@/lib/types'

const fetcher = (url: string) => fetch(url).then(res => res.json())

const CATEGORY_ICONS: Record<string, React.ReactNode> = {
  sources: <BookMarked className="h-3.5 w-3.5" />,
  concepts: <Layers className="h-3.5 w-3.5" />,
  procedures: <Wrench className="h-3.5 w-3.5" />,
  components: <Box className="h-3.5 w-3.5" />,
  images: <ImageIcon className="h-3.5 w-3.5" />,
}

const CATEGORY_LABELS: Record<string, string> = {
  sources: 'Fonti',
  concepts: 'Concetti',
  procedures: 'Procedure',
  components: 'Componenti',
  images: 'Immagini',
}

export default function WikiViewer() {
  const [selectedPage, setSelectedPage] = useState<WikiPage | null>(null)
  const [pageContent, setPageContent] = useState<string>('')
  const [isIngesting, setIsIngesting] = useState(false)
  const [isLinting, setIsLinting] = useState(false)
  const [lintResult, setLintResult] = useState<any>(null)
  const [ingestResult, setIngestResult] = useState<any>(null)
  const [showLearnPanel, setShowLearnPanel] = useState(false)
  const [learnText, setLearnText] = useState('')
  const [learnTitle, setLearnTitle] = useState('')
  const [isLearning, setIsLearning] = useState(false)
  const [learnResult, setLearnResult] = useState<any>(null)

  const { data: wikiData, isLoading } = useSWR<WikiStatus>(
    '/api/wiki',
    fetcher,
    { refreshInterval: isIngesting ? 3000 : 0 }
  )

  const loadPage = useCallback(async (page: WikiPage) => {
    setSelectedPage(page)
    setLintResult(null)
    setIngestResult(null)
    try {
      const res = await fetch(`/api/wiki/pages/${page.category}/${page.name}`)
      const data = await res.json()
      setPageContent(data.content || 'Contenuto non disponibile.')
    } catch {
      setPageContent('Errore nel caricamento della pagina.')
    }
  }, [])

  const handleIngest = async () => {
    if (!confirm('Vuoi compilare/aggiornare la wiki con tutti i documenti caricati?')) return
    setIsIngesting(true)
    setIngestResult(null)
    setLintResult(null)
    setSelectedPage(null)
    try {
      const res = await fetch('/api/wiki/ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      })
      const data = await res.json()
      setIngestResult(data)
      mutate('/api/wiki')
    } catch (err) {
      setIngestResult({ success: false, error: 'Errore di rete' })
    } finally {
      setIsIngesting(false)
    }
  }

  const handleLint = async () => {
    setIsLinting(true)
    setLintResult(null)
    setIngestResult(null)
    setSelectedPage(null)
    try {
      const res = await fetch('/api/wiki/lint', { method: 'POST' })
      const data = await res.json()
      setLintResult(data)
    } catch {
      setLintResult({ success: false, error: 'Errore di rete' })
    } finally {
      setIsLinting(false)
    }
  }

  const handleLearn = async () => {
    if (!learnText.trim()) return
    setIsLearning(true)
    setLearnResult(null)
    try {
      const res = await fetch('/api/wiki/learn', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: learnText, hint_title: learnTitle.trim() || undefined }),
      })
      const data = await res.json()
      setLearnResult(data)
      if (data.success) {
        setLearnText('')
        setLearnTitle('')
        mutate('/api/wiki')
      }
    } catch {
      setLearnResult({ success: false, error: 'Errore di rete' })
    } finally {
      setIsLearning(false)
    }
  }

  const handleReset = async () => {
    if (!confirm('Sei sicuro di voler eliminare TUTTA la wiki? Le pagine generate verranno cancellate.')) return
    try {
      await fetch('/api/wiki', { method: 'DELETE' })
      setSelectedPage(null)
      setPageContent('')
      setLintResult(null)
      setIngestResult(null)
      mutate('/api/wiki')
    } catch (err) {
      console.error('Wiki reset error:', err)
    }
  }

  const pages = wikiData?.pages || []
  const categories = wikiData?.categories || { sources: 0, concepts: 0, procedures: 0, components: 0, images: 0 }
  const totalPages = wikiData?.total_pages || 0

  // Raggruppa pagine per categoria
  const pagesByCategory: Record<string, WikiPage[]> = {}
  for (const page of pages) {
    if (!pagesByCategory[page.category]) pagesByCategory[page.category] = []
    pagesByCategory[page.category].push(page)
  }

  return (
    <div className="flex-1 flex min-h-0 overflow-hidden">
      {/* Sidebar Wiki */}
      <div className="w-72 border-r border-border bg-card flex flex-col min-h-0">
        {/* Header */}
        <div className="shrink-0 p-3 border-b border-border">
          <div className="flex items-center gap-2 mb-2">
            <BookOpen className="h-4 w-4 text-primary" />
            <h2 className="font-semibold text-sm">LLM Wiki</h2>
            <span className="ml-auto text-xs text-muted-foreground bg-muted px-1.5 py-0.5 rounded">
              {totalPages} pagine
            </span>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 gap-1 text-[11px] mb-3">
            {Object.entries(CATEGORY_LABELS).map(([key, label]) => (
              <div key={key} className="flex items-center gap-1 text-muted-foreground">
                {CATEGORY_ICONS[key]}
                <span>{label}: <strong className="text-foreground">{categories[key as keyof typeof categories] || 0}</strong></span>
              </div>
            ))}
          </div>

          {/* Actions */}
          <div className="flex gap-1.5">
            <Button
              size="sm"
              variant="default"
              className="flex-1 text-xs h-7"
              onClick={handleIngest}
              disabled={isIngesting || isLinting}
            >
              {isIngesting ? (
                <RefreshCw className="h-3 w-3 mr-1 animate-spin" />
              ) : (
                <Upload className="h-3 w-3 mr-1" />
              )}
              {isIngesting ? 'Compilazione...' : 'Ingest'}
            </Button>
            <Button
              size="sm"
              variant="outline"
              className="text-xs h-7"
              onClick={handleLint}
              disabled={isIngesting || isLinting || totalPages === 0}
            >
              {isLinting ? (
                <RefreshCw className="h-3 w-3 mr-1 animate-spin" />
              ) : (
                <Search className="h-3 w-3 mr-1" />
              )}
              Lint
            </Button>
            <Button
              size="sm"
              variant="ghost"
              className="text-xs h-7 text-muted-foreground hover:text-destructive"
              onClick={handleReset}
              disabled={isIngesting || isLinting || totalPages === 0}
            >
              <Trash2 className="h-3 w-3" />
            </Button>
          </div>
        </div>

        {/* Page List */}
        <div className="flex-1 overflow-y-auto min-h-0 p-2">
          {isLoading ? (
            <div className="text-xs text-muted-foreground p-3 text-center">Caricamento...</div>
          ) : totalPages === 0 ? (
            <div className="text-xs text-muted-foreground p-3 text-center">
              <BookOpen className="h-8 w-8 mx-auto mb-2 opacity-30" />
              <p>Wiki vuota</p>
              <p className="mt-1">Carica documenti e premi &quot;Ingest&quot;</p>
            </div>
          ) : (
            Object.entries(CATEGORY_LABELS).map(([cat, label]) => {
              const catPages = pagesByCategory[cat] || []
              if (catPages.length === 0) return null
              return (
                <div key={cat} className="mb-3">
                  <div className="flex items-center gap-1.5 text-[11px] font-semibold text-muted-foreground uppercase tracking-wider px-1 mb-1">
                    {CATEGORY_ICONS[cat]}
                    {label} ({catPages.length})
                  </div>
                  <div className="space-y-0.5">
                    {catPages.map(page => (
                      <button
                        key={page.path}
                        onClick={() => loadPage(page)}
                        className={`w-full text-left text-xs px-2 py-1.5 rounded transition-colors flex items-center gap-1.5 ${
                          selectedPage?.path === page.path
                            ? 'bg-primary/10 text-primary font-medium'
                            : 'text-foreground hover:bg-muted'
                        }`}
                      >
                        <FileText className="h-3 w-3 shrink-0 opacity-50" />
                        <span className="truncate">{page.title}</span>
                        {page.links_count > 0 && (
                          <span className="ml-auto text-[10px] text-muted-foreground">{page.links_count}↗</span>
                        )}
                      </button>
                    ))}
                  </div>
                </div>
              )
            })
          )}
        </div>
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-y-auto min-h-0 p-6">
        {ingestResult && (
          <div className="mb-6 p-4 rounded-lg border bg-card">
            <h3 className="font-semibold text-sm mb-2 flex items-center gap-2">
              <Upload className="h-4 w-4 text-primary" />
              Risultato Ingest
            </h3>
            {ingestResult.success ? (
              <div className="text-sm space-y-1">
                <p>Documenti processati: <strong>{ingestResult.documents_processed}</strong></p>
                <p>Pagine create: <strong>{ingestResult.total_pages_created}</strong></p>
                <p>Pagine aggiornate: <strong>{ingestResult.total_pages_updated}</strong></p>
                {ingestResult.results?.map((r: any, i: number) => (
                  <div key={i} className="text-xs text-muted-foreground mt-1">
                    {r.success ? '✅' : '❌'} {r.filename}
                    {r.error && <span className="text-destructive ml-1">— {r.error}</span>}
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-destructive">{ingestResult.error || 'Errore durante ingest'}</p>
            )}
          </div>
        )}

        {lintResult && (
          <div className="mb-6 p-4 rounded-lg border bg-card">
            <h3 className="font-semibold text-sm mb-2 flex items-center gap-2">
              <Search className="h-4 w-4 text-primary" />
              Audit Wiki — Health Score: {lintResult.health_score}%
            </h3>
            {lintResult.issues?.length > 0 ? (
              <div className="space-y-2 mt-2">
                {lintResult.issues.map((issue: any, i: number) => (
                  <div key={i} className="flex items-start gap-2 text-xs">
                    <AlertTriangle className={`h-3.5 w-3.5 shrink-0 mt-0.5 ${
                      issue.severity === 'high' ? 'text-destructive' :
                      issue.severity === 'medium' ? 'text-yellow-500' : 'text-muted-foreground'
                    }`} />
                    <div>
                      <span className="font-medium">[{issue.type}]</span>{' '}
                      <span className="text-muted-foreground">{issue.page}</span>
                      <p className="mt-0.5">{issue.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-green-600">✅ Nessun problema trovato!</p>
            )}
            {lintResult.suggestions?.length > 0 && (
              <div className="mt-3 pt-3 border-t space-y-1">
                <p className="text-xs font-semibold text-muted-foreground">Suggerimenti:</p>
                {lintResult.suggestions.map((s: string, i: number) => (
                  <p key={i} className="text-xs text-muted-foreground">• {s}</p>
                ))}
              </div>
            )}
          </div>
        )}

        {selectedPage ? (
          <div>
            <div className="flex items-center gap-2 mb-4">
              <span className="text-xs text-muted-foreground bg-muted px-2 py-0.5 rounded">
                {CATEGORY_LABELS[selectedPage.category] || selectedPage.category}
              </span>
              <ChevronRight className="h-3 w-3 text-muted-foreground" />
              <h2 className="font-semibold">{selectedPage.title}</h2>
            </div>

            {/* Anteprima immagine per pagine della categoria images */}
            {selectedPage.category === 'images' && (() => {
              const match = pageContent.match(/`images\/([^`]+)`/)
              const imgFile = match?.[1]
              if (!imgFile) return null
              const imgUrl = `/images/${imgFile}`
              return (
                <div className="mb-5">
                  <Dialog>
                    <DialogTrigger asChild>
                      <div className="group relative inline-block cursor-zoom-in rounded-lg overflow-hidden border border-border hover:border-primary/60 transition-colors max-w-full">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={imgUrl}
                          alt={imgFile}
                          className="max-h-64 max-w-full object-contain block"
                        />
                        <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity bg-black/40 rounded-lg">
                          <ZoomIn className="h-8 w-8 text-white" />
                        </div>
                      </div>
                    </DialogTrigger>
                    <DialogContent className="max-w-4xl max-h-[90vh] flex flex-col">
                      <DialogHeader>
                        <DialogTitle className="text-sm font-mono">{imgFile}</DialogTitle>
                      </DialogHeader>
                      <div className="flex-1 overflow-auto flex items-center justify-center">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img src={imgUrl} alt={imgFile} className="max-w-full max-h-full object-contain" />
                      </div>
                    </DialogContent>
                  </Dialog>
                </div>
              )
            })()}

            <div className="prose prose-sm dark:prose-invert max-w-none">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{pageContent}</ReactMarkdown>
            </div>
            <div className="mt-6 pt-4 border-t text-xs text-muted-foreground">
              <p>Ultima modifica: {new Date(selectedPage.modified).toLocaleString('it-IT')}</p>
              {selectedPage.links.length > 0 && (
                <p className="mt-1">
                  Link: {selectedPage.links.map(l => `[[${l}]]`).join(', ')}
                </p>
              )}
            </div>
          </div>
        ) : !ingestResult && !lintResult ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <BookOpen className="h-16 w-16 text-muted-foreground/20 mb-4" />
            <h2 className="text-xl font-semibold text-foreground mb-2">LLM Wiki</h2>
            <p className="text-muted-foreground max-w-md text-sm mb-4">
              La wiki compila i tuoi documenti in pagine strutturate e interconnesse.
              Invece di cercare nei chunk raw, il LLM legge pagine pre-sintetizzate.
            </p>
            <div className="text-xs text-muted-foreground space-y-1 max-w-sm text-left">
              <p><strong>1. Ingest</strong> — Carica documenti e premi &quot;Ingest&quot; per compilare la wiki</p>
              <p><strong>2. Query</strong> — Usa la tab Chat Wiki per fare domande sulla wiki</p>
              <p><strong>3. Lint</strong> — Premi &quot;Lint&quot; per verificare la salute della wiki</p>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  )
}
