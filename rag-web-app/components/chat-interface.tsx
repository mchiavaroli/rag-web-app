'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import useSWR, { mutate } from 'swr'
import { Send, Trash2, Bot, User, BarChart2, History, ScrollText } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { ChatMessage, Source, ModelProvider } from '@/lib/types'
import { SourceViewer } from './source-viewer'

const fetcher = (url: string) => fetch(url).then(res => res.json())

type DisplayMessage = ChatMessage & { optimistic?: boolean }

const QUICK_COMMANDS = [
  { label: 'Logs', command: 'logs', icon: ScrollText },
  { label: 'Statistiche', command: 'stats', icon: BarChart2 },
  { label: 'Cronologia', command: 'history', icon: History },
]

/** Deduplica le fonti PDF (stessa sorgente → una riga sola) e lascia tutte le immagini */
function deduplicateSources(sources: Source[]): Source[] {
  const seen = new Set<string>()
  return sources.filter(s => {
    if (s.type === 'image') return true
    const key = s.title || s.path
    if (seen.has(key)) return false
    seen.add(key)
    return true
  })
}

function ChatInterface() {
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [pendingUserMsg, setPendingUserMsg] = useState<DisplayMessage | null>(null)
  const [models, setModels] = useState<ModelProvider[]>([])
  const [activeModel, setActiveModel] = useState<string>('')
  const activeModelRef = useRef<string>('')  // ← aggiungi questo
  const [defaultModel, setDefaultModel] = useState<string>('')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const { data } = useSWR<{ messages: ChatMessage[] }>(
    '/api/chat',
    fetcher,
    { refreshInterval: 0 }
  )

  useEffect(() => {
  fetch('/api/models')
    .then(res => res.json())
    .then((data) => {
      setModels(data.models || [])
      setDefaultModel(data.default || '')
      setActiveModel(data.default || '')  // data.default è già la chiave del dict
      activeModelRef.current = data.default || ''
    })
    .catch(() => setModels([]))
}, [])

  const serverMessages = data?.messages || []
  const displayMessages: DisplayMessage[] = pendingUserMsg
    ? [...serverMessages, pendingUserMsg]
    : serverMessages

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => { scrollToBottom() }, [displayMessages, isLoading, scrollToBottom])

  const sendMessage = async (text: string) => {
    const trimmed = text.trim()
    if (!trimmed || isLoading) return

    const optimistic: DisplayMessage = {
      id: `opt_${Date.now()}`,
      role: 'user',
      content: trimmed,
      timestamp: new Date().toISOString(),
      optimistic: true,
    }

    setInput('')
    setIsLoading(true)
    setPendingUserMsg(optimistic)

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: trimmed, model: activeModelRef.current }),
      })
      if (!response.ok) throw new Error('Errore nella risposta')
      await mutate('/api/chat')
    } catch (err) {
      console.error('Chat error:', err)
    } finally {
      setPendingUserMsg(null)
      setIsLoading(false)
      textareaRef.current?.focus()
    }
  }

  const handleSubmit = (e?: React.FormEvent) => {
    e?.preventDefault()
    sendMessage(input)
  }

  const handleClearChat = async () => {
    if (!confirm('Sei sicuro di voler eliminare tutta la cronologia?')) return
    try {
      await fetch('/api/chat', { method: 'DELETE' })
      mutate('/api/chat')
    } catch (err) {
      console.error('Clear chat error:', err)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  return (
    <div className="flex-1 flex flex-col min-h-0 bg-background overflow-hidden">
      {/* Header + Model Switcher */}
      <div className="shrink-0 px-4 py-3 border-b border-border flex items-center justify-between bg-card">
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-full bg-primary flex items-center justify-center shrink-0">
            <Bot className="h-5 w-5 text-primary-foreground" />
          </div>
          <div>
            <h1 className="font-semibold text-card-foreground leading-tight">RAG Assistant</h1>
            <p className="text-xs text-muted-foreground">Fai domande sui tuoi documenti</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {models.length > 1 && (
            <select
              className="text-xs border rounded px-2 py-1 bg-background text-foreground"
              value={activeModel}
              onChange={e => {
                setActiveModel(e.target.value)
                activeModelRef.current = e.target.value
              }}
              disabled={isLoading}
              title="Seleziona modello LLM"
            >
              {models.map(m => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
            </select>
          )}
          {serverMessages.length > 0 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleClearChat}
              className="text-muted-foreground hover:text-destructive shrink-0"
            >
              <Trash2 className="h-4 w-4 mr-1.5" />
              Pulisci
            </Button>
          )}
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto min-h-0 px-4 py-4 space-y-5">
        {displayMessages.length === 0 && !isLoading ? (
          <div className="flex flex-col items-center justify-center h-full text-center py-16">
            <div className="h-16 w-16 rounded-full bg-muted flex items-center justify-center mb-4">
              <Bot className="h-8 w-8 text-muted-foreground" />
            </div>
            <h2 className="text-xl font-semibold text-foreground mb-2">Benvenuto!</h2>
            <p className="text-muted-foreground max-w-sm text-sm">
              Carica dei PDF nella barra laterale e inizia a fare domande.
            </p>
          </div>
        ) : (
          displayMessages.map(message => (
            <div
              key={message.id}
              className={`flex items-end gap-2 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {message.role === 'assistant' && (
                <div className="h-7 w-7 rounded-full bg-primary flex items-center justify-center shrink-0 mb-1">
                  <Bot className="h-3.5 w-3.5 text-primary-foreground" />
                </div>
              )}

              <div className={`flex flex-col gap-1 min-w-0 ${
                message.role === 'user' ? 'items-end max-w-[75%]' : 'items-start max-w-[82%]'
              }`}>
                <div className={`rounded-2xl px-4 py-3 w-full break-words ${
                  message.role === 'user'
                    ? 'bg-primary text-primary-foreground rounded-br-sm'
                    : 'bg-card border border-border rounded-bl-sm'
                } ${message.optimistic ? 'opacity-70' : ''}`}>

                  <MarkdownContent content={message.content} isUser={message.role === 'user'} />

                  {message.sources && message.sources.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-border/40">
                      <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
                        Fonti ({deduplicateSources(message.sources).length})
                      </p>
                      {message.sources.filter(s => s.type === 'image').length > 0 && (
                        <div className="flex flex-wrap gap-2 mb-2">
                          {message.sources
                            .filter(s => s.type === 'image')
                            .map((source, idx) => (
                              <SourceViewer key={`img-${idx}`} source={source} />
                            ))}
                        </div>
                      )}
                      <div className="space-y-1">
                        {deduplicateSources(message.sources)
                          .filter(s => s.type === 'pdf')
                          .map((source, idx) => (
                            <SourceViewer key={`pdf-${idx}`} source={source} />
                          ))}
                      </div>
                    </div>
                  )}
                </div>

                <span className="text-[10px] text-muted-foreground px-1">
                  {new Date(message.timestamp).toLocaleTimeString('it-IT', { hour: '2-digit', minute: '2-digit' })}
                </span>
              </div>

              {message.role === 'user' && (
                <div className="h-7 w-7 rounded-full bg-secondary flex items-center justify-center shrink-0 mb-1">
                  <User className="h-3.5 w-3.5 text-secondary-foreground" />
                </div>
              )}
            </div>
          ))
        )}

        {isLoading && (
          <div className="flex items-end gap-2 justify-start">
            <div className="h-7 w-7 rounded-full bg-primary flex items-center justify-center shrink-0 mb-1">
              <Bot className="h-3.5 w-3.5 text-primary-foreground" />
            </div>
            <div className="bg-card border border-border rounded-2xl rounded-bl-sm px-4 py-3">
              <div className="flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full bg-muted-foreground/60 animate-bounce [animation-delay:0ms]" />
                <span className="h-2 w-2 rounded-full bg-muted-foreground/60 animate-bounce [animation-delay:150ms]" />
                <span className="h-2 w-2 rounded-full bg-muted-foreground/60 animate-bounce [animation-delay:300ms]" />
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Quick commands */}
      <div className="shrink-0 px-4 pt-2 flex gap-2 bg-background">
        {QUICK_COMMANDS.map(({ label, command, icon: Icon }) => (
          <button
            key={command}
            onClick={() => sendMessage(command)}
            disabled={isLoading}
            className="flex items-center gap-1.5 px-3 py-1 rounded-full border border-border bg-card hover:bg-accent text-xs text-muted-foreground hover:text-foreground transition-colors disabled:opacity-40"
          >
            <Icon className="h-3 w-3" />
            {label}
          </button>
        ))}
      </div>

      {/* Input */}
      <div className="shrink-0 px-4 pt-2 pb-3 bg-card border-t border-border mt-2">
        <form onSubmit={handleSubmit} className="flex gap-2 items-end">
          <Textarea
            ref={textareaRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Scrivi una domanda sui tuoi documenti..."
            className="min-h-[44px] max-h-32 resize-none flex-1"
            disabled={isLoading}
            rows={1}
          />
          <Button
            type="submit"
            size="icon"
            disabled={!input.trim() || isLoading}
            className="h-[44px] w-[44px] shrink-0"
          >
            <Send className="h-4 w-4" />
          </Button>
        </form>
        <p className="text-xs text-muted-foreground mt-1.5 text-center">
          Invio per inviare · Shift+Invio per andare a capo
        </p>
      </div>
    </div>
  )
}

export default ChatInterface

/** Componente markdown con stili Tailwind */
function MarkdownContent({ content, isUser }: { content: string; isUser: boolean }) {
  const prose = isUser ? 'text-primary-foreground' : 'text-foreground'

  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        h1: ({ children }) => <h1 className={`text-base font-bold mt-3 mb-1 ${prose}`}>{children}</h1>,
        h2: ({ children }) => <h2 className={`text-sm font-bold mt-2 mb-1 ${prose}`}>{children}</h2>,
        h3: ({ children }) => <h3 className={`text-sm font-semibold mt-2 mb-0.5 ${prose}`}>{children}</h3>,
        p: ({ children }) => <p className={`text-sm leading-relaxed mb-1 ${prose}`}>{children}</p>,
        strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
        em: ({ children }) => <em className="italic">{children}</em>,
        code: ({ children, className }) => {
          const isBlock = className?.includes('language-')
          return isBlock
            ? <code className={`block bg-black/10 rounded px-2 py-1 text-xs font-mono my-1 whitespace-pre-wrap ${prose}`}>{children}</code>
            : <code className="bg-black/10 rounded px-1 text-xs font-mono">{children}</code>
        },
        pre: ({ children }) => <pre className="my-2 overflow-x-auto">{children}</pre>,
        ul: ({ children }) => <ul className={`list-disc list-inside text-sm space-y-0.5 mb-1 pl-2 ${prose}`}>{children}</ul>,
        ol: ({ children }) => <ol className={`list-decimal list-inside text-sm space-y-0.5 mb-1 pl-2 ${prose}`}>{children}</ol>,
        li: ({ children }) => <li className="leading-relaxed">{children}</li>,
        blockquote: ({ children }) => (
          <blockquote className={`border-l-2 border-current/30 pl-3 italic my-1 opacity-80 ${prose}`}>{children}</blockquote>
        ),
        hr: () => <hr className="my-2 border-current/20" />,
        table: ({ children }) => (
          <div className="overflow-x-auto my-2">
            <table className="text-xs border-collapse w-full">{children}</table>
          </div>
        ),
        th: ({ children }) => <th className="border border-current/20 px-2 py-1 font-semibold bg-black/5 text-left">{children}</th>,
        td: ({ children }) => <td className="border border-current/20 px-2 py-1">{children}</td>,
        a: ({ href, children }) => (
          <a href={href} target="_blank" rel="noopener noreferrer" className="underline opacity-80 hover:opacity-100">{children}</a>
        ),
      }}
    >
      {content}
    </ReactMarkdown>
  )
}
