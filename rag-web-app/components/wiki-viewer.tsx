'use client'

import { useState, useCallback, useMemo } from 'react'
import useSWR, { mutate } from 'swr'
import {
  BookOpen, FileText, Upload, Search, AlertTriangle,
  RefreshCw, Trash2, ChevronRight, Layers, Wrench, Box, BookMarked, ImageIcon, ZoomIn, GraduationCap, Link2, Share2
} from 'lucide-react'
import WikiGraph from '@/components/wiki-graph'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { Textarea } from '@/components/ui/textarea'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { WikiStatus, WikiPage } from '@/lib/types'

const fetcher = (url: string) => fetch(url).then(res => res.json())

/** Converte [[slug]] in link markdown navigabili */
function processWikiLinks(content: string, pages: WikiPage[]): string {
  return content.replace(/\[\[([^\]]+)\]\]/g, (_, slug) => {
    const page = pages.find(p => p.name === slug + '.md' || p.name === slug)
    const title = page?.title ?? slug.replace(/-/g, ' ').replace(/\b\w/g, (c: string) => c.toUpperCase())
    return `[${title}](#wiki-${slug})`
  })
}

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
  const [viewMode, setViewMode] = useState<'pages' | 'graph'>('pages')

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

  // Custom ReactMarkdown components
  const markdownComponents = useMemo(() => ({
    // Render p as div to avoid invalid nesting when code blocks appear inside paragraphs
    p: ({ children }: any) => <div className="my-2">{children}</div>,
    a: ({ href, children }: any) => {
      if (href?.startsWith('#wiki-')) {
        const slug = href.slice(6)
        const target = pages.find(p => p.name === slug + '.md' || p.name === slug)
        return (
          <button
            type="button"
            onClick={() => target && loadPage(target)}
            className="inline-flex items-center gap-0.5 text-primary hover:underline underline-offset-2 font-medium cursor-pointer"
          >
            <Link2 className="h-3 w-3 shrink-0" />
            {children}
          </button>
        )
      }
      return (
        <a href={href} target="_blank" rel="noopener noreferrer" className="text-primary hover:underline underline-offset-2 break-all">
          {children}
        </a>
      )
    },
    // In react-markdown v10 the `inline` prop was removed from `code`.
    // Block code is wrapped in a `pre` by react-markdown; we style that here.
    pre: ({ children }: any) => (
      <pre className="bg-muted p-3 rounded-md overflow-x-auto text-xs font-mono my-2">{children}</pre>
    ),
    code: ({ className, children }: any) => {
      const raw = String(children ?? '')
      const text = raw.trim()
      // Inline code: no language className and no newlines
      const isInline = !className && !raw.includes('\n')
      if (isInline && /^images\/.+\.(png|jpe?g|gif|webp|svg)$/i.test(text)) {
        const imgUrl = `/${text}`
        const fname = text.replace('images/', '')
        // Calcola lo slug della pagina wiki immagine (senza estensione, _ → -)
        const imgSlug = fname.replace(/\.[^.]+$/, '').replace(/_/g, '-')
        const imgWikiPage = pages.find(p =>
          p.category === 'images' && (p.name === imgSlug + '.md' || p.name === imgSlug)
        )
        // Se siamo già sulla pagina di questa immagine, non mostrare miniatura ridondante
        if (imgWikiPage && imgWikiPage.name === selectedPage?.name) {
          return <code className="bg-muted px-1.5 py-0.5 rounded text-[0.82em] font-mono">{children}</code>
        }
        return (
          <span className="inline-flex items-center gap-1.5 align-middle mx-1 my-0.5">
            <Dialog>
              <DialogTrigger asChild>
                <span className="group relative inline-block cursor-zoom-in rounded overflow-hidden border border-border hover:border-primary/60 transition-colors">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={imgUrl}
                    alt={fname}
                    className="h-16 max-w-[120px] object-cover block"
                    onError={e => { (e.currentTarget as HTMLImageElement).style.display = 'none' }}
                  />
                  <span className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity bg-black/40">
                    <ZoomIn className="h-4 w-4 text-white" />
                  </span>
                </span>
              </DialogTrigger>
              <DialogContent className="max-w-4xl max-h-[90vh] flex flex-col">
                <DialogHeader>
                  <DialogTitle className="text-sm font-mono">{fname}</DialogTitle>
                </DialogHeader>
                <div className="flex-1 overflow-auto flex items-center justify-center">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={imgUrl} alt={fname} className="max-w-full max-h-full object-contain" />
                </div>
              </DialogContent>
            </Dialog>
            {imgWikiPage && imgWikiPage.name !== selectedPage?.name && (
              <button
                type="button"
                onClick={() => loadPage(imgWikiPage)}
                className="inline-flex items-center gap-0.5 text-[11px] text-primary hover:underline underline-offset-2 font-medium cursor-pointer"
                title={`Apri pagina wiki: ${imgWikiPage.title}`}
              >
                <Link2 className="h-3 w-3 shrink-0" />
                {imgWikiPage.title}
              </button>
            )}
          </span>
        )
      }
      return <code className={isInline ? 'bg-muted px-1.5 py-0.5 rounded text-[0.82em] font-mono' : className}>{children}</code>
    },
    blockquote: ({ children }: any) => (
      <blockquote className="border-l-4 border-primary/50 bg-primary/5 pl-4 pr-2 py-0.5 rounded-r-md my-3 text-muted-foreground italic">
        {children}
      </blockquote>
    ),
    h2: ({ children }: any) => (
      <h2 className="text-sm font-semibold mt-5 mb-2 text-foreground border-b border-border pb-1 flex items-center gap-1.5">
        {children}
      </h2>
    ),
    hr: () => <hr className="my-4 border-border" />,
    table: ({ children }: any) => (
      <div className="overflow-x-auto my-3 rounded-md border border-border">
        <table className="text-sm border-collapse w-full">{children}</table>
      </div>
    ),
    th: ({ children }: any) => (
      <th className="border-b border-border bg-muted px-3 py-1.5 text-left font-semibold text-xs">{children}</th>
    ),
    td: ({ children }: any) => (
      <td className="border-b border-border/50 px-3 py-1.5 text-sm last:border-b-0">{children}</td>
    ),
  }), [pages, loadPage, selectedPage])

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
              variant={showLearnPanel ? 'secondary' : 'outline'}
              className="text-xs h-7"
              onClick={() => { setShowLearnPanel(v => !v); setLearnResult(null) }}
              disabled={isIngesting || isLinting || isLearning}
              title="Insegna un nuovo concetto alla wiki"
            >
              <GraduationCap className="h-3 w-3 mr-1" />
              Insegna
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

          {/* Learn Panel */}
          {showLearnPanel && (
            <div className="mt-3 pt-3 border-t border-border space-y-2">
              <p className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-1">
                <GraduationCap className="h-3 w-3" />
                Insegna alla Wiki
              </p>
              <input
                type="text"
                placeholder="Titolo (opzionale)"
                value={learnTitle}
                onChange={e => setLearnTitle(e.target.value)}
                className="w-full text-xs rounded-md border border-input bg-background px-2 py-1.5 placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
                disabled={isLearning}
              />
              <Textarea
                placeholder="Scrivi in linguaggio naturale il concetto, procedura o componente da aggiungere alla wiki..."
                value={learnText}
                onChange={e => setLearnText(e.target.value)}
                className="text-xs min-h-[90px] max-h-48 resize-y"
                disabled={isLearning}
              />
              <Button
                size="sm"
                variant="default"
                className="w-full text-xs h-7"
                onClick={handleLearn}
                disabled={isLearning || !learnText.trim()}
              >
                {isLearning ? (
                  <RefreshCw className="h-3 w-3 mr-1 animate-spin" />
                ) : (
                  <GraduationCap className="h-3 w-3 mr-1" />
                )}
                {isLearning ? 'Elaborazione...' : 'Aggiungi alla Wiki'}
              </Button>
              {learnResult && (
                <div className={`text-[11px] rounded-md px-2 py-1.5 border ${learnResult.success ? 'bg-green-50 dark:bg-green-950 border-green-200 dark:border-green-800 text-green-700 dark:text-green-300' : 'bg-red-50 dark:bg-red-950 border-red-200 dark:border-red-800 text-red-700 dark:text-red-300'}`}>
                  {learnResult.success ? (
                    <>
                      ✅ {learnResult.log_entry}
                      <br />
                      <span className="opacity-70">{learnResult.pages_created} create, {learnResult.pages_updated} aggiornate</span>
                    </>
                  ) : (
                    <>❌ {learnResult.error}</>
                  )}
                </div>
              )}
            </div>
          )}
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
      <div className="flex-1 flex flex-col min-h-0">
        {/* View toggle */}
        <div className="shrink-0 flex items-center gap-1 px-4 pt-3 pb-1 border-b border-border">
          <button
            type="button"
            onClick={() => setViewMode('pages')}
            className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-md transition-colors ${
              viewMode === 'pages'
                ? 'bg-primary/10 text-primary font-medium'
                : 'text-muted-foreground hover:bg-muted'
            }`}
          >
            <FileText className="h-3.5 w-3.5" />
            Pagine
          </button>
          <button
            type="button"
            onClick={() => setViewMode('graph')}
            className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-md transition-colors ${
              viewMode === 'graph'
                ? 'bg-primary/10 text-primary font-medium'
                : 'text-muted-foreground hover:bg-muted'
            }`}
          >
            <Share2 className="h-3.5 w-3.5" />
            Grafo
          </button>
        </div>

        {/* Graph view */}
        {viewMode === 'graph' && (
          <div className="flex-1 min-h-0">
            <WikiGraph pages={pages} onSelectPage={(page) => { setViewMode('pages'); loadPage(page) }} />
          </div>
        )}

        {/* Pages view */}
        {viewMode === 'pages' && <div className="flex-1 overflow-y-auto min-h-0 p-6">
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

            <div className="prose prose-sm dark:prose-invert max-w-none [&_p]:my-2 [&_ul]:my-2 [&_ol]:my-2 [&_li]:my-0.5">
              <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                {processWikiLinks(pageContent, pages)}
              </ReactMarkdown>
            </div>
            <div className="mt-6 pt-4 border-t">
              <p className="text-xs text-muted-foreground mb-2">
                Ultima modifica: {new Date(selectedPage.modified).toLocaleString('it-IT')}
              </p>
              {selectedPage.links.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {selectedPage.links.map(slug => {
                    const target = pages.find(p => p.name === slug + '.md' || p.name === slug)
                    return (
                      <button
                        key={slug}
                        type="button"
                        onClick={() => target && loadPage(target)}
                        className="inline-flex items-center gap-1 text-[11px] px-2 py-0.5 rounded-full border border-border bg-muted hover:bg-primary/10 hover:border-primary/40 hover:text-primary transition-colors"
                      >
                        <Link2 className="h-2.5 w-2.5" />
                        {target?.title ?? slug}
                      </button>
                    )
                  })}
                </div>
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
        </div>}
      </div>
    </div>
  )
}
