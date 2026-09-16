import { useState } from 'react'
import type { FormEvent } from 'react'
import { sendChatMessage } from './api'
import type { ChatMessage } from './types'
import './App.css'

function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [emotionStreak, setEmotionStreak] = useState(0)
  const [stuckState, setStuckState] = useState(0)
  const [mood, setMood] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const currentMessage = input.trim()
    if (!currentMessage || isLoading) return

    setError(null)
    setInput('')
    setIsLoading(true)

    try {
      const response = await sendChatMessage({
        current_message: currentMessage,
        messages,
        emotion_streak: emotionStreak,
        stuck_state: stuckState,
      })

      setMessages((current) => [
        ...current,
        { role: 'user', content: currentMessage },
        { role: 'assistant', content: response.assistant_response },
      ])
      setEmotionStreak(response.emotion_streak)
      setStuckState(response.stuck_state)
      setMood(response.mood)
    } catch (requestError) {
      setInput(currentMessage)
      setError(requestError instanceof Error ? requestError.message : 'The conversation service is temporarily unavailable.')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <main className="app-shell">
      <header className="app-header">
        <div className="brand-lockup">
          <span className="brand-mark" aria-hidden="true" />
          <div><p className="eyebrow">A quiet place to think</p><h1>Therapy AI</h1></div>
        </div>
        <p className="boundary">AI wellness companion <span aria-hidden="true">•</span> Not a therapist or emergency service</p>
      </header>

      <div className="workspace">
        <section className="conversation-panel" aria-label="Conversation">
          {messages.length === 0 ? (
            <div className="empty-state">
              <span className="empty-orbit" aria-hidden="true">✦</span>
              <h2>What's on your mind today?</h2>
              <p>You can talk things through, reflect, or ask for a little direction.</p>
              <div className="starter-chips" aria-label="Conversation ideas"><span>Talk it out</span><span>Help me decide</span><span>Check in</span><span>I'm feeling stressed</span></div>
            </div>
          ) : (
            <div className="message-list" aria-live="polite">
              {messages.map((message, index) => (
                <article className={`message-row ${message.role}`} key={`${message.role}-${index}`}>
                  <div className="message-bubble"><span className="message-label">{message.role === 'assistant' ? 'Therapy AI' : 'You'}</span><p>{message.content}</p></div>
                </article>
              ))}
              {isLoading && <div className="message-row assistant" aria-label="Therapy AI is responding"><div className="message-bubble loading-bubble"><span className="message-label">Therapy AI</span><span className="loading-dots"><i /><i /><i /></span></div></div>}
            </div>
          )}
        </section>

        <aside className="status-panel" aria-label="Conversation details">
          <div className="status-heading"><span className="status-dot" aria-hidden="true" /><span>Present moment</span></div>
          <p className="status-copy">Your conversation stays in this browser for now.</p>
          <div className="status-divider" />
          <dl className="status-list"><div><dt>Current mood</dt><dd>{mood ?? 'Not yet noticed'}</dd></div><div><dt>Messages</dt><dd>{messages.length}</dd></div></dl>
        </aside>
      </div>

      <div className="composer-wrap">
        {error && <p className="error-message" role="alert">{error}</p>}
        <form className="composer" onSubmit={handleSubmit}>
          <label className="sr-only" htmlFor="message-input">Share what's on your mind</label>
          <textarea id="message-input" value={input} onChange={(event) => setInput(event.target.value)} placeholder="Share what's on your mind..." rows={1} disabled={isLoading} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); event.currentTarget.form?.requestSubmit() } }} />
          <button type="submit" disabled={isLoading || !input.trim()} aria-label="Send message"><span aria-hidden="true">↑</span></button>
        </form>
        <p className="composer-hint">Press Enter to send <span aria-hidden="true">·</span> Shift + Enter for a new line</p>
      </div>
    </main>
  )
}

export default App
