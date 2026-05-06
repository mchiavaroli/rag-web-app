import { NextResponse } from 'next/server'

const RAG_BACKEND_URL = process.env.RAG_BACKEND_URL || 'http://localhost:8000'

// GET /api/wiki/graph — Nodi e archi della wiki per il grafo
export async function GET() {
  try {
    const res = await fetch(`${RAG_BACKEND_URL}/api/wiki/graph`)
    const data = await res.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error('Wiki graph error:', error)
    return NextResponse.json(
      { error: 'Errore nel recupero del grafo wiki' },
      { status: 500 }
    )
  }
}
