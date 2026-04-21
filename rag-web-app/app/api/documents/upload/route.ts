import { NextRequest, NextResponse } from 'next/server'

const RAG_BACKEND_URL = process.env.RAG_BACKEND_URL || 'http://localhost:8000'

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const files = formData.getAll('files') as File[]

    if (!files || files.length === 0) {
      return NextResponse.json({ error: 'Nessun file caricato' }, { status: 400 })
    }

    const uploaded = []
    const errors = []

    for (const file of files) {
      if (file.type !== 'application/pdf') {
        errors.push(`${file.name}: non è un PDF`)
        continue
      }

      // Invia il file al backend FastAPI
      const backendForm = new FormData()
      backendForm.append('file', file)

      try {
        const response = await fetch(`${RAG_BACKEND_URL}/api/documents`, {
          method: 'POST',
          body: backendForm,
        })

        if (!response.ok) {
          const detail = await response.json().catch(() => ({}))
          errors.push(`${file.name}: ${detail.detail || 'errore backend'}`)
          continue
        }

        const data = await response.json()
        uploaded.push(data.document)
      } catch (err) {
        errors.push(`${file.name}: impossibile contattare il backend`)
      }
    }

    if (uploaded.length === 0) {
      return NextResponse.json(
        { error: errors.join('; ') || 'Nessun file caricato' },
        { status: 400 }
      )
    }

    return NextResponse.json({
      success: true,
      documents: uploaded,
      message: `${uploaded.length} documento/i caricato/i. Indicizzazione avviata.`,
      errors: errors.length > 0 ? errors : undefined,
    })
  } catch (error) {
    console.error('Upload error:', error)
    return NextResponse.json({ error: 'Errore durante il caricamento' }, { status: 500 })
  }
}
