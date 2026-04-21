import { NextResponse } from 'next/server'

const RAG_BACKEND_URL = process.env.RAG_BACKEND_URL || 'http://localhost:8000'

export async function GET() {
  // Chiama il backend per ottenere la lista dei modelli disponibili
  const response = await fetch(`${RAG_BACKEND_URL}/api/models`)
  if (!response.ok) {
    return NextResponse.json({ error: 'Errore recupero modelli' }, { status: 500 })
  }
  const models = await response.json()
  return NextResponse.json(models)
}
