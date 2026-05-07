import { NextRequest, NextResponse } from 'next/server'

const RAG_BACKEND_URL = process.env.RAG_BACKEND_URL || 'http://localhost:8000'

// POST /api/wiki/fix — Applica le correzioni suggerite dall'audit Lint
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()

    const res = await fetch(`${RAG_BACKEND_URL}/api/wiki/fix`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })

    if (!res.ok) {
      const detail = await res.json().catch(() => ({}))
      throw new Error(detail.detail || `Backend error: ${res.status}`)
    }

    const data = await res.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error('Wiki fix error:', error)
    const msg = error instanceof Error ? error.message : 'Errore sconosciuto'
    return NextResponse.json({ success: false, error: msg }, { status: 500 })
  }
}
