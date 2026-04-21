import { NextRequest, NextResponse } from 'next/server'

const RAG_BACKEND_URL = process.env.RAG_BACKEND_URL || 'http://localhost:8000'

/** GET /api/index/status — Stato corrente dell'indice RAG */
export async function GET() {
  try {
    const response = await fetch(`${RAG_BACKEND_URL}/api/status`, {
      cache: 'no-store',
    })
    if (!response.ok) throw new Error(`Backend error: ${response.status}`)
    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error('Error fetching index status:', error)
    return NextResponse.json(
      {
        status: 'error',
        message: 'Backend non raggiungibile. Avvia il server RAG con: python api_server.py',
        total_chunks: 0,
        text_chunks: 0,
        image_chunks: 0,
        documents: [],
      },
      { status: 503 }
    )
  }
}

/** POST /api/index/status — Forza la re-indicizzazione */
export async function POST(_request: NextRequest) {
  try {
    const response = await fetch(`${RAG_BACKEND_URL}/api/index`, {
      method: 'POST',
    })
    if (!response.ok) throw new Error(`Backend error: ${response.status}`)
    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error('Error triggering index:', error)
    return NextResponse.json(
      { error: 'Impossibile avviare l\'indicizzazione. Backend non raggiungibile.' },
      { status: 503 }
    )
  }
}
