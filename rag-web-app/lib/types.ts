export interface Document {
  id: string
  name: string
  size: number
  uploadedAt: string
  path: string
  indexed?: boolean
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  timestamp: string
}

export interface Source {
  type: 'pdf' | 'image'
  path: string
  title?: string
  page?: number
  preview?: string
}

export interface RAGResponse {
  answer: string
  sources: Source[]
}

export interface IndexStatus {
  status: 'idle' | 'building' | 'ready' | 'error'
  message: string
  started_at?: string
  completed_at?: string
  total_chunks: number
  text_chunks: number
  image_chunks: number
  documents: Document[]
}
