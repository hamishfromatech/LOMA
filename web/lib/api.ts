import type {
  Conversation,
  ConversationSummary,
  McpOverview,
  MemoryPayload,
  ModelsPayload,
  OnboardingResult,
  OnboardingState,
  SettingsPayload,
  Status,
  StreamEvent,
} from "./types"

// ---------------------------------------------------------------------------
// Storage helpers (wrappers around localStorage with graceful fallback)
// ---------------------------------------------------------------------------

const KEY_LAST_CONVERSATION = "loma:last_conversation"
const KEY_DRAFT = "loma:draft"

export function getLastConversation(): string | null {
  try {
    const v = localStorage.getItem(KEY_LAST_CONVERSATION)
    return v || null
  } catch {
    return null
  }
}

export function rememberConversation(id: string): void {
  try {
    localStorage.setItem(KEY_LAST_CONVERSATION, id)
  } catch {
    /* localStorage full or blocked */
  }
}

export function getDraft(): string {
  try {
    return localStorage.getItem(KEY_DRAFT) ?? ""
  } catch {
    return ""
  }
}

export function setDraft(text: string): void {
  try {
    if (text) {
      localStorage.setItem(KEY_DRAFT, text)
    } else {
      localStorage.removeItem(KEY_DRAFT)
    }
  } catch {
    /* localStorage full or blocked */
  }
}

export const API_BASE =
  process.env.NEXT_PUBLIC_LOMA_API ?? "http://127.0.0.1:5050"

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init)
  if (!response.ok) {
    try {
      return (await response.json()) as T
    } catch {
      // HTML / traceback: cap ~200 chars instead of rendering raw
      const detail = await response.text()
      throw new Error(
        `${response.status} ${response.statusText}: ${(detail || "").slice(0, 200)}`,
      )
    }
  }
  return (await response.json()) as T
}

export function getStatus(): Promise<Status> {
  return json<Status>("/api/status")
}

/**
 * Models installed locally, discovered by the API from Ollama's own listing
 * (`/api/tags`` with a `/v1/models` fallback). Refetched whenever the picker
 * opens, so a `ollama pull` shows up without a page reload.
 */
export function listModels(): Promise<ModelsPayload> {
  return json<ModelsPayload>("/api/models")
}

export function resetConversation(): Promise<{ ok: boolean }> {
  return json("/api/reset", { method: "POST" })
}

export async function transcribe(blob: Blob): Promise<string> {
  const form = new FormData()
  form.append("audio", blob, "clip.webm")
  const result = await json<{ text: string }>("/api/transcribe", {
    method: "POST",
    body: form,
  })
  return result.text
}

export async function speak(text: string): Promise<Blob> {
  const response = await fetch(`${API_BASE}/api/speak`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  })
  if (!response.ok) throw new Error("speech failed")
  return response.blob()
}

/**
 * POST /api/chat and read Server-Sent Events off the response body.
 *
 * `EventSource` cannot POST, so we parse the stream by hand: chunks arrive as
 * `data: {...}\n\n` records.
 */
export async function streamChat(
  message: string,
  onEvent: (event: StreamEvent) => void,
  options: {
    conversationId: string
    speak?: boolean
    signal?: AbortSignal
    /** Data URLs ("data:image/png;base64,…"); the API strips the prefix. */
    images?: string[]
  },
): Promise<void> {
  const response = await fetch(
    `${API_BASE}/api/conversations/${encodeURIComponent(options.conversationId)}/chat`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        speak: options.speak ?? true,
        images: options.images?.length ? options.images : undefined,
      }),
      signal: options.signal,
    },
  )

  if (!response.ok || !response.body) {
    throw new Error(`chat failed: ${response.status}`)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""

  while (true) {
    const { done, value } = await reader.read()
    if (done) {
      // Flush: a trailing record without \n\n is a split UTF-8 byte;
      // decode with stream:false forces a final parse of leftover bytes.
      const tail = decoder.decode(value, { stream: false })
      if (tail.trim()) {
        const payload = tail.trim().replace(/^data:\s*/, "")
        if (payload) {
          try { onEvent(JSON.parse(payload) as StreamEvent) }
          catch { /* partial — not worth tearing the stream */ }
        }
      }
      break
    }
    buffer += decoder.decode(value, { stream: true })

    let boundary = buffer.indexOf("\n\n")
    while (boundary !== -1) {
      const record = buffer.slice(0, boundary).trim()
      buffer = buffer.slice(boundary + 2)
      boundary = buffer.indexOf("\n\n")
      if (!record.startsWith("data:")) continue
      const payload = record.slice(5).trim()
      if (!payload) continue
      try {
        onEvent(JSON.parse(payload) as StreamEvent)
      } catch {
        // A partial or unknown record is not worth tearing the stream down.
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

export function getSettings(): Promise<SettingsPayload> {
  return json<SettingsPayload>("/api/settings")
}

export function updateSettings(patch: {
  llm_model?: string
  vlm_model?: string
  tts_voice?: string
  temperature?: number
  speak_replies?: boolean
}): Promise<SettingsPayload> {
  return json<SettingsPayload>("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  })
}

export function getMcp(): Promise<McpOverview> {
  return json<McpOverview>("/api/mcp")
}

export function reloadMcp(): Promise<McpOverview> {
  return json<McpOverview>("/api/mcp/reload", { method: "POST" })
}

export function addMcpServer(payload: {
  name: string
  command?: string
  args?: string
  env?: string
  url?: string
}): Promise<McpOverview> {
  return json<McpOverview>("/api/mcp/servers", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })
}

export async function removeMcpServer(name: string): Promise<McpOverview> {
  const response = await fetch(
    `${API_BASE}/api/mcp/servers/${encodeURIComponent(name)}`,
    { method: "DELETE" },
  )
  if (!response.ok) throw new Error(await response.text())
  return (await response.json()) as McpOverview
}

export async function previewVoice(voice: string): Promise<Blob> {
  const response = await fetch(`${API_BASE}/api/speak`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: "Hi, I am LOMA. This is my voice.", voice }),
  })
  if (!response.ok) throw new Error("preview failed")
  return response.blob()
}

// ---------------------------------------------------------------------------
// Conversations
// ---------------------------------------------------------------------------

export function listConversations(): Promise<ConversationSummary[]> {
  return json<ConversationSummary[]>("/api/conversations")
}

export function createConversation(title = "New chat"): Promise<Conversation> {
  return json<Conversation>("/api/conversations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  })
}

export function getConversation(id: string): Promise<Conversation> {
  return json<Conversation>(`/api/conversations/${encodeURIComponent(id)}`)
}

export async function renameConversation(
  id: string,
  title: string,
): Promise<Conversation> {
  const response = await fetch(
    `${API_BASE}/api/conversations/${encodeURIComponent(id)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    },
  )
  if (!response.ok) throw new Error(await response.text())
  return (await response.json()) as Conversation
}

export async function deleteConversation(id: string): Promise<void> {
  const response = await fetch(
    `${API_BASE}/api/conversations/${encodeURIComponent(id)}`,
    { method: "DELETE" },
  )
  if (!response.ok) throw new Error(await response.text())
}

export async function stopConversation(id: string): Promise<void> {
  await fetch(
    `${API_BASE}/api/conversations/${encodeURIComponent(id)}/stop`,
    { method: "POST" },
  )
}

// ---------------------------------------------------------------------------
// First run
// ---------------------------------------------------------------------------

export function getOnboarding(): Promise<OnboardingState> {
  return json<OnboardingState>("/api/onboarding")
}

export function completeOnboarding(answers: {
  name: string
  role: string
  goals: string[]
  style: string
  notes: string
}): Promise<OnboardingResult> {
  return json<OnboardingResult>("/api/onboarding", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(answers),
  })
}

export function skipOnboarding(): Promise<OnboardingResult> {
  return json<OnboardingResult>("/api/onboarding/skip", { method: "POST" })
}

export function resetOnboarding(): Promise<OnboardingState> {
  return json<OnboardingState>("/api/onboarding/reset", { method: "POST" })
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

export function getMemory(): Promise<MemoryPayload> {
  return json<MemoryPayload>("/api/memory")
}

export function writeMemory(content: string): Promise<MemoryPayload> {
  return json<MemoryPayload>("/api/memory", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content }),
  })
}

export function clearMemory(): Promise<MemoryPayload> {
  return json<MemoryPayload>("/api/memory", { method: "DELETE" })
}
