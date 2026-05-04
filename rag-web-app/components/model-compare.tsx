'use client'

import { useState, useEffect } from 'react'
import { Bot, Send, Zap } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { ModelProvider, Source } from '@/lib/types'
import { SourceViewer } from './source-viewer'

interface CompareResult {
  model: string
  answer: string
  sources: Source[]
  latency_ms: number
}

interface CompareState {
  loading: boolean
  results: [CompareResult | null, CompareResult | null]
  error: string | null
}

function deduplicatePdfSources(sources: Source[]): Source[] {
  const seen = new Set<string>()
  return sources.filter(s => {
    if (s.type === 'image') return true
    const key = s.title || s.path
    if (seen.has(key)) return false
    seen.add(key)
    return true
  })
}

export default function ModelCompare() {
  const [query, setQuery] = useState('')
  const [modelA, setModelA] = useState('')
  const [modelB, setModelB] = useState('')
  const [models, setModels] = useState<ModelProvider[]>([])
  const [state, setState] = useState<CompareState>({
    loading: false,
    results: [null, null],
    error: null,
  })

  useEffect(() => {
    fetch('/api/models')
      .then(res => res.json())
      .then((data: { models: ModelProvider[]; default: string }) => {
        const list = data.models || []
        setModels(list)
        if (list.length > 0) setModelA(list[0].id)
        if (list.length > 1) setModelB(list[1].id)
        else if (list.length > 0) setModelB(list[0].id)
      })
      .catch(() => setModels([]))
  }, [])

  const modelName = (id: string) => models.find(m => m.id === id)?.name || id

  const runCompare = async () => {
    if (!query.trim() || !modelA || !modelB || state.loading) return
    setState({ loading: true, results: [null, null], error: null })

    try {
      const response = await fetch('/api/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query.trim(), model_a: modelA, model_b: modelB }),
      })

      if (!response.ok) {
        const err = await response.json().catch(() => ({})) as { error?: string }
        throw new Error(err.error || `Errore: ${response.status}`)
      }

      const data = await response.json() as { results: [CompareResult, CompareResult] }
      setState({ loading: false, results: [data.results[0], data.results[1]], error: null })
    } catch (err) {
      setState({
        loading: false,
        results: [null, null],
        error: err instanceof Error ? err.message : 'Errore sconosciuto',
      })
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      runCompare()
    }
  }

  const formatLatency = (ms: number) =>
    ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`

  return (
    <div className="flex-1 flex flex-col min-h-0 bg-background overflow-hidden">
      {/* Header */}
      <div className="shrink-0 px-4 py-3 border-b border-border bg-card flex items-center gap-3">
        <div className="h-9 w-9 rounded-full bg-primary flex items-center justify-center shrink-0">
          <Zap className="h-5 w-5 text-primary-foreground" />
        </div>
        <div>
          <h1 className="font-semibold text-card-foreground leading-tight">Confronto Modelli</h1>
          <p className="text-xs text-muted-foreground">Stessa query, due modelli in parallelo</p>
        </div>
      </div>

      {/* Controls */}
      <div className="shrink-0 px-4 py-3 border-b border-border bg-muted/30 space-y-3">
        <div className="grid grid-cols-2 gap-4">
          {[
            { label: 'Modello A', value: modelA, onChange: setModelA },
            { label: 'Modello B', value: modelB, onChange: setModelB },
          ].map(({ label, value, onChange }) => (
            <div key={label} className="flex items-center gap-2">
              <span className="text-xs font-medium text-muted-foreground w-20 shrink-0">{label}</span>
              <select
                className="flex-1 text-xs border rounded px-2 py-1.5 bg-background text-foreground"
                value={value}
                onChange={e => onChange(e.target.value)}
                disabled={state.loading}
              >
                {models.map(m => (
                  <option key={m.id} value={m.id}>{m.name}</option>
                ))}
              </select>
            </div>
          ))}
        </div>

        <div className="flex gap-2">
          <Textarea
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Inserisci la domanda da confrontare tra i due modelli... (Invio per inviare)"
            className="min-h-[60px] max-h-[120px] resize-none text-sm"
            disabled={state.loading}
          />
          <Button
            onClick={runCompare}
            disabled={!query.trim() || !modelA || !modelB || state.loading}
            className="shrink-0 self-end"
          >
            {state.loading ? (
              <div className="h-4 w-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
          </Button>
        </div>
      </div>

      {/* Side-by-side results */}
      <div className="flex-1 min-h-0 grid grid-cols-2 divide-x divide-border overflow-hidden">
        {([0, 1] as const).map(i => {
          const result = state.results[i]
          const modelId = i === 0 ? modelA : modelB
          const label = i === 0 ? 'A' : 'B'
          const pdfs = result ? deduplicatePdfSources(result.sources).filter(s => s.type === 'pdf') : []
          const images = result ? result.sources.filter(s => s.type === 'image') : []

          return (
            <div key={i} className="flex flex-col min-h-0 overflow-hidden">
              {/* Column header */}
              <div className="shrink-0 px-3 py-2 bg-muted/20 border-b border-border flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="h-5 w-5 rounded-full bg-primary text-primary-foreground text-xs flex items-center justify-center font-bold shrink-0">
                    {label}
                  </span>
                  <span className="text-xs font-medium text-foreground truncate">{modelName(modelId)}</span>
                </div>
                {result && (
                  <span className="text-xs text-muted-foreground bg-muted rounded px-1.5 py-0.5 shrink-0 ml-2">
                    {formatLatency(result.latency_ms)}
                  </span>
                )}
              </div>

              {/* Column content */}
              <div className="flex-1 overflow-y-auto px-4 py-4">
                {state.loading && !result && (
                  <div className="flex items-center gap-2 text-muted-foreground text-sm">
                    <div className="h-3 w-3 border-2 border-current border-t-transparent rounded-full animate-spin shrink-0" />
                    <span>Elaborazione in corso...</span>
                  </div>
                )}

                {state.error && !result && (
                  <div className="text-sm text-destructive bg-destructive/10 rounded px-3 py-2">
                    {state.error}
                  </div>
                )}

                {result && (
                  <div className="space-y-4">
                    <div className="prose prose-sm max-w-none dark:prose-invert text-sm leading-relaxed">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {result.answer}
                      </ReactMarkdown>
                    </div>

                    {result.sources.length > 0 && (
                      <div className="pt-3 border-t border-border/40">
                        <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
                          Fonti ({(pdfs.length + images.length)})
                        </p>
                        {images.length > 0 && (
                          <div className="flex flex-wrap gap-2 mb-2">
                            {images.map((source, idx) => (
                              <SourceViewer key={`img-${idx}`} source={source} />
                            ))}
                          </div>
                        )}
                        {pdfs.length > 0 && (
                          <div className="space-y-1">
                            {pdfs.map((source, idx) => (
                              <SourceViewer key={`pdf-${idx}`} source={source} />
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}

                {!state.loading && !result && !state.error && (
                  <div className="flex flex-col items-center justify-center h-full text-center py-16 text-muted-foreground">
                    <Bot className="h-10 w-10 mb-3 opacity-30" />
                    <p className="text-sm">La risposta apparirà qui</p>
                  </div>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
