import { NextRequest, NextResponse } from 'next/server'
import { writeFile, readFile, mkdir } from 'fs/promises'
import { existsSync } from 'fs'
import path from 'path'
import type { ChatMessage, RAGResponse } from '@/lib/types'

const CHAT_HISTORY_FILE = path.join(process.cwd(), 'data', 'chat-history.json')
const SESSION_FILE = path.join(process.cwd(), 'data', 'session.json')
const RAG_BACKEND_URL = process.env.RAG_BACKEND_URL || 'http://localhost:8000'

async function ensureDataDir() {
  const dataDir = path.join(process.cwd(), 'data')
  if (!existsSync(dataDir)) {
    await mkdir(dataDir, { recursive: true })
  }
}

async function getOrCreateSessionId(): Promise<string> {
  try {
    if (existsSync(SESSION_FILE)) {
      const data = await readFile(SESSION_FILE, 'utf-8')
      const parsed = JSON.parse(data)
      if (parsed.sessionId) return parsed.sessionId
    }
  } catch { /* ignora errori, crea nuova sessione */ }
  const sessionId = `web_sess_${Date.now()}`
  await writeFile(SESSION_FILE, JSON.stringify({ sessionId }))
  return sessionId
}

async function resetSessionId(): Promise<void> {
  const sessionId = `web_sess_${Date.now()}`
  await writeFile(SESSION_FILE, JSON.stringify({ sessionId }))
}

async function getChatHistory(): Promise<ChatMessage[]> {
  try {
    if (!existsSync(CHAT_HISTORY_FILE)) return []
    const data = await readFile(CHAT_HISTORY_FILE, 'utf-8')
    return JSON.parse(data)
  } catch {
    return []
  }
}

async function saveChatHistory(messages: ChatMessage[]) {
  await ensureDataDir()
  await writeFile(CHAT_HISTORY_FILE, JSON.stringify(messages, null, 2))
}

/** Converte i path relativi del backend in URL assoluti servibili dal browser. */
function resolveBackendPaths(ragResponse: RAGResponse): RAGResponse {
  if (!ragResponse.sources) return ragResponse
  return {
    ...ragResponse,
    sources: ragResponse.sources.map(source => ({
      ...source,
      path: source.path.startsWith('http')
        ? source.path
        : `${RAG_BACKEND_URL}${source.path}`,
    })),
  }
}

export async function POST(request: NextRequest) {
  try {
    await ensureDataDir()

    const { message } = await request.json()
    if (!message || typeof message !== 'string') {
      return NextResponse.json({ error: 'Messaggio non valido' }, { status: 400 })
    }

    const history = await getChatHistory()
    const sessionId = await getOrCreateSessionId()

    const userMessage: ChatMessage = {
      id: `msg_${Date.now()}_user`,
      role: 'user',
      content: message,
      timestamp: new Date().toISOString(),
    }
    history.push(userMessage)

    let ragResponse: RAGResponse

    try {
      const response = await fetch(`${RAG_BACKEND_URL}/api/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: message, session_id: sessionId }),
      })

      if (!response.ok) {
        const detail = await response.json().catch(() => ({}))
        throw new Error(detail.detail || `Backend error: ${response.status}`)
      }

      ragResponse = resolveBackendPaths(await response.json())
    } catch (error) {
      console.error('RAG backend error:', error)
      const msg = error instanceof Error ? error.message : 'Errore sconosciuto'
      ragResponse = {
        answer: `Mi dispiace, non ho potuto ottenere una risposta dal backend RAG. Dettaglio: ${msg}`,
        sources: [],
      }
    }

    const assistantMessage: ChatMessage = {
      id: `msg_${Date.now()}_assistant`,
      role: 'assistant',
      content: ragResponse.answer,
      sources: ragResponse.sources,
      timestamp: new Date().toISOString(),
    }
    history.push(assistantMessage)
    await saveChatHistory(history)

    return NextResponse.json({ success: true, message: assistantMessage })
  } catch (error) {
    console.error('Chat route error:', error)
    return NextResponse.json(
      { error: "Errore durante l'elaborazione del messaggio" },
      { status: 500 }
    )
  }
}

export async function GET() {
  try {
    const history = await getChatHistory()
    return NextResponse.json({ messages: history })
  } catch (error) {
    console.error('Error fetching chat history:', error)
    return NextResponse.json(
      { error: 'Errore nel recupero della cronologia' },
      { status: 500 }
    )
  }
}

export async function DELETE() {
  try {
    await ensureDataDir()
    await writeFile(CHAT_HISTORY_FILE, JSON.stringify([]))
    await resetSessionId()
    
    // Elimina anche i log dal backend
    try {
      await fetch(`${RAG_BACKEND_URL}/api/logs`, { method: 'DELETE' })
    } catch (e) {
      console.warn('Impossibile eliminare log dal backend:', e)
    }
    
    return NextResponse.json({ success: true, message: 'Cronologia chat e log eliminati' })
  } catch (error) {
    console.error('Error clearing chat:', error)
    return NextResponse.json(
      { error: "Errore durante l'eliminazione della cronologia" },
      { status: 500 }
    )
  }
}
