import { NextRequest, NextResponse } from 'next/server'

const RAG_BACKEND_URL = process.env.RAG_BACKEND_URL || 'http://localhost:8000'

export async function GET() {
  try {
    const response = await fetch(`${RAG_BACKEND_URL}/api/documents`, {
      cache: 'no-store',
    })
    if (!response.ok) throw new Error(`Backend error: ${response.status}`)
    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error('Error fetching documents:', error)
    return NextResponse.json(
      { error: 'Impossibile contattare il backend RAG. Verifica che sia in esecuzione.' },
      { status: 503 }
    )
  }
}

export async function DELETE(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const id = searchParams.get('id')
    if (!id) {
      return NextResponse.json({ error: 'ID documento mancante' }, { status: 400 })
    }

    // L'id nel backend corrisponde al nome del file
    const response = await fetch(`${RAG_BACKEND_URL}/api/documents/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    })
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}))
      throw new Error(detail.detail || `Backend error: ${response.status}`)
    }

    return NextResponse.json({ success: true, message: 'Documento eliminato con successo' })
  } catch (error) {
    console.error('Delete error:', error)
    return NextResponse.json(
      { error: 'Errore durante l\'eliminazione' },
      { status: 500 }
    )
  }
}
