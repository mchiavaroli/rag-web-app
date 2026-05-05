import { NextRequest, NextResponse } from 'next/server'
import { readFile } from 'fs/promises'
import { existsSync } from 'fs'
import path from 'path'
import type { RAGResponse } from '@/lib/types'

const RAG_BACKEND_URL = process.env.RAG_BACKEND_URL || 'http://localhost:8000'
const SESSION_FILE = path.join(process.cwd(), 'data', 'session.json')

async function getSessionId(): Promise<string> {
  try {
    if (existsSync(SESSION_FILE)) {
      const data = await readFile(SESSION_FILE, 'utf-8')
      const parsed = JSON.parse(data)
      if (parsed.sessionId) return parsed.sessionId
    }
  } catch { /* */ }
  return `web_sess_${Date.now()}`
}

/** Converte i path relativi backend in URL assoluti servibili dal browser. */
function resolveBackendPaths(data: RAGResponse): RAGResponse {
  if (!data.sources) return data
  return {
    ...data,
    sources: data.sources.map(source => ({
      ...source,
      path: source.type === 'wiki'
        ? source.path  // le pagine wiki non hanno un URL backend
        : source.path.startsWith('http')
          ? source.path
          : `${RAG_BACKEND_URL}${source.path}`,
    })),
  }
}

// POST /api/wiki/query — Query wiki
export async function POST(request: NextRequest) {
  try {
    const { message, model } = await request.json()
    if (!message || typeof message !== 'string') {
      return NextResponse.json({ error: 'Messaggio non valido' }, { status: 400 })
    }

    const sessionId = await getSessionId()

    const res = await fetch(`${RAG_BACKEND_URL}/api/wiki/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: message, session_id: sessionId, model }),
    })

    if (!res.ok) {
      const detail = await res.json().catch(() => ({}))
      throw new Error(detail.detail || `Backend error: ${res.status}`)
    }

    const rawData: RAGResponse = await res.json()
    const data = resolveBackendPaths(rawData)

    return NextResponse.json({
      success: true,
      message: {
        id: `msg_${Date.now()}_wiki`,
        role: 'assistant',
        content: data.answer,
        sources: data.sources || [],
        timestamp: new Date().toISOString(),
      },
    })
  } catch (error) {
    console.error('Wiki query error:', error)
    const msg = error instanceof Error ? error.message : 'Errore sconosciuto'
    return NextResponse.json(
      {
        success: true,
        message: {
          id: `msg_${Date.now()}_wiki_error`,
          role: 'assistant',
          content: `Errore nella query wiki: ${msg}`,
          sources: [],
          timestamp: new Date().toISOString(),
        },
      }
    )
  }
}
