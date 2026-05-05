'use client'

import { useState } from 'react'
import { MessageSquare, BookOpen } from 'lucide-react'
import { DocumentSidebar } from '@/components/document-sidebar'
import ChatInterface from '@/components/chat-interface'
import WikiViewer from '@/components/wiki-viewer'
import type { ChatMessage } from '@/lib/types'

type Mode = 'wiki' | 'wiki-chat'

export default function Home() {
  const [mode, setMode] = useState<Mode>('wiki-chat')
  const [wikiMessages, setWikiMessages] = useState<ChatMessage[]>([])

  return (
    <main className="h-screen flex overflow-hidden">
      <DocumentSidebar />
      <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
        {/* Mode toggle bar */}
        <div className="shrink-0 border-b border-border bg-muted/20 flex items-center px-3 py-1.5 gap-1">
          <button
            onClick={() => setMode('wiki-chat')}
            className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-md transition-colors ${
              mode === 'wiki-chat'
                ? 'bg-background text-foreground shadow-sm font-medium border border-border'
                : 'text-muted-foreground hover:text-foreground hover:bg-background/60'
            }`}
          >
            <MessageSquare className="h-3.5 w-3.5" />
            Chat Wiki
          </button>
          <button
            onClick={() => setMode('wiki')}
            className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-md transition-colors ${
              mode === 'wiki'
                ? 'bg-background text-foreground shadow-sm font-medium border border-border'
                : 'text-muted-foreground hover:text-foreground hover:bg-background/60'
            }`}
          >
            <BookOpen className="h-3.5 w-3.5" />
            Wiki
          </button>
        </div>

        {/* Mantieni i componenti montati con hidden per preservare lo stato al cambio tab */}
        <div className={mode === 'wiki-chat' ? 'flex flex-col flex-1 min-h-0' : 'hidden'}>
          <ChatInterface wikiMode wikiMessages={wikiMessages} setWikiMessages={setWikiMessages} />
        </div>
        <div className={mode === 'wiki' ? 'flex flex-col flex-1 min-h-0' : 'hidden'}>
          <WikiViewer />
        </div>
      </div>
    </main>
  )
}
