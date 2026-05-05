import { NextRequest, NextResponse } from 'next/server'

const RAG_BACKEND_URL = process.env.RAG_BACKEND_URL || 'http://localhost:8000'

// GET /api/wiki — Stato wiki + lista pagine
export async function GET() {
  try {
    const [statusRes, pagesRes] = await Promise.all([
      fetch(`${RAG_BACKEND_URL}/api/wiki/status`),
      fetch(`${RAG_BACKEND_URL}/api/wiki/pages`),
    ])

    const status = await statusRes.json()
    const pages = await pagesRes.json()

    return NextResponse.json({ ...status, pages: pages.pages || [] })
  } catch (error) {
    console.error('Wiki status error:', error)
    return NextResponse.json(
      { error: 'Errore nel recupero dello stato wiki' },
      { status: 500 }
    )
  }
}

// DELETE /api/wiki — Reset wiki
export async function DELETE() {
  try {
    const res = await fetch(`${RAG_BACKEND_URL}/api/wiki`, { method: 'DELETE' })
    const data = await res.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error('Wiki reset error:', error)
    return NextResponse.json(
      { error: 'Errore nel reset della wiki' },
      { status: 500 }
    )
  }
}
