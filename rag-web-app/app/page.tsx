'use client'

import { useState } from 'react'
import { MessageSquare, Zap } from 'lucide-react'
import { DocumentSidebar } from '@/components/document-sidebar'
import ChatInterface from '@/components/chat-interface'
import ModelCompare from '@/components/model-compare'

type Mode = 'chat' | 'compare'

export default function Home() {
  const [mode, setMode] = useState<Mode>('chat')

  return (
    <main className="h-screen flex overflow-hidden">
      <DocumentSidebar />
      <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
        {/* Mode toggle bar */}
        <div className="shrink-0 border-b border-border bg-muted/20 flex items-center px-3 py-1.5 gap-1">
          <button
            onClick={() => setMode('chat')}
            className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-md transition-colors ${
              mode === 'chat'
                ? 'bg-background text-foreground shadow-sm font-medium border border-border'
                : 'text-muted-foreground hover:text-foreground hover:bg-background/60'
            }`}
          >
            <MessageSquare className="h-3.5 w-3.5" />
            Chat
          </button>
          <button
            onClick={() => setMode('compare')}
            className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-md transition-colors ${
              mode === 'compare'
                ? 'bg-background text-foreground shadow-sm font-medium border border-border'
                : 'text-muted-foreground hover:text-foreground hover:bg-background/60'
            }`}
          >
            <Zap className="h-3.5 w-3.5" />
            Confronto
          </button>
        </div>
        {mode === 'chat' ? <ChatInterface /> : <ModelCompare />}
      </div>
    </main>
  )
}
