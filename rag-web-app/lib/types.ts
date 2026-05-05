
export interface ModelProvider {
  id: string        
  name: string
  provider: string
  deployment_name: string
  endpoint: string
  max_tokens?: number
  max_completion_tokens?: number
  temperature: number
  api_version?: string
}
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
  type: 'pdf' | 'image' | 'wiki'
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

// ============================================================
// WIKI TYPES
// ============================================================

export interface WikiPage {
  name: string
  category: string
  title: string
  path: string
  content?: string
  size: number
  modified: string
  links_count: number
  links: string[]
}

export interface WikiStatus {
  status: 'ready' | 'empty'
  total_pages: number
  categories: {
    sources: number
    concepts: number
    procedures: number
    components: number
  }
  has_index: boolean
  pages: WikiPage[]
}

export interface WikiLintIssue {
  type: 'orphan' | 'broken_link' | 'contradiction' | 'missing_page' | 'incomplete'
  severity: 'low' | 'medium' | 'high'
  page: string
  description: string
}

export interface WikiLintResult {
  success: boolean
  issues: WikiLintIssue[]
  suggestions: string[]
  health_score: number
  stats: Omit<WikiStatus, 'pages'>
}

export interface WikiIngestResult {
  success: boolean
  documents_processed: number
  total_pages_created: number
  total_pages_updated: number
  results: Array<{
    filename: string
    success: boolean
    pages_created?: number
    pages_updated?: number
    error?: string
  }>
}
