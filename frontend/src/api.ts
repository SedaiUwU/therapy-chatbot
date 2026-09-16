import type { ChatRequest, ChatResponse } from './types'

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')
const GENERIC_ERROR = 'The conversation service is temporarily unavailable. Please try again.'

async function readError(response: Response): Promise<string> {
  if (response.status === 422) return 'Please enter a message before sending.'
  return GENERIC_ERROR
}

export async function sendChatMessage(request: ChatRequest): Promise<ChatResponse> {
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    })
  } catch {
    throw new Error(GENERIC_ERROR)
  }

  if (!response.ok) throw new Error(await readError(response))

  try {
    return (await response.json()) as ChatResponse
  } catch {
    throw new Error(GENERIC_ERROR)
  }
}

export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/health`)
    return response.ok
  } catch {
    return false
  }
}
