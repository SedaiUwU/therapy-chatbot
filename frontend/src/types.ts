export type ChatRole = 'user' | 'assistant'

export interface ChatMessage {
  role: ChatRole
  content: string
}

export interface ChatRequest {
  current_message: string
  messages: ChatMessage[]
  emotion_streak: number
  stuck_state: number
}

export interface ChatResponse {
  assistant_response: string
  mood: string | null
  emotion_streak: number
  stuck_state: number
  safety_category: string | null
}
