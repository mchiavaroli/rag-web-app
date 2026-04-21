import { DocumentSidebar } from '@/components/document-sidebar'
import { ChatInterface } from '@/components/chat-interface'

export default function Home() {
  return (
    <main className="h-screen flex overflow-hidden">
      <DocumentSidebar />
      <ChatInterface />
    </main>
  )
}
