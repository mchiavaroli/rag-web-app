import { NextRequest, NextResponse } from 'next/server'

const RAG_BACKEND_URL = process.env.RAG_BACKEND_URL || 'http://localhost:8000'

function resolveBackendPaths(result: { sources?: Array<{ path: string; [key: string]: unknown }>; [key: string]: unknown }) {
  if (!result.sources) return result
  return {
    ...result,
    sources: result.sources.map(source => ({
      ...source,
      path: source.path.startsWith('http') ? source.path : `${RAG_BACKEND_URL}${source.path}`,
    })),
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { query, session_id, model_a, model_b } = body

    if (!query || typeof query !== 'string' || !model_a || !model_b) {
      return NextResponse.json({ error: 'Parametri mancanti: query, model_a, model_b' }, { status: 400 })
    }

    const response = await fetch(`${RAG_BACKEND_URL}/api/compare`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, session_id, model_a, model_b }),
    })

    if (!response.ok) {
      const detail = await response.json().catch(() => ({}))
      return NextResponse.json(
        { error: (detail as { detail?: string }).detail || `Backend error: ${response.status}` },
        { status: response.status }
      )
    }

    const data = await response.json() as { results: Array<{ sources?: Array<{ path: string; [key: string]: unknown }>; [key: string]: unknown }> }
    return NextResponse.json({
      results: data.results.map(resolveBackendPaths),
    })
  } catch (error) {
    const msg = error instanceof Error ? error.message : 'Errore sconosciuto'
    return NextResponse.json({ error: msg }, { status: 500 })
  }
}
