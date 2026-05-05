import { NextRequest, NextResponse } from 'next/server'

const RAG_BACKEND_URL = process.env.RAG_BACKEND_URL || 'http://localhost:8000'

// GET /api/wiki/pages/[category]/[filename] — Leggi pagina wiki
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ category: string; filename: string }> }
) {
  try {
    const { category, filename } = await params

    const res = await fetch(
      `${RAG_BACKEND_URL}/api/wiki/pages/${encodeURIComponent(category)}/${encodeURIComponent(filename)}`
    )

    if (!res.ok) {
      if (res.status === 404) {
        return NextResponse.json({ error: 'Pagina non trovata' }, { status: 404 })
      }
      throw new Error(`Backend error: ${res.status}`)
    }

    const data = await res.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error('Wiki page error:', error)
    return NextResponse.json(
      { error: 'Errore nel recupero della pagina wiki' },
      { status: 500 }
    )
  }
}
