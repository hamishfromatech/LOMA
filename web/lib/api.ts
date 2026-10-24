import type {
  Conversation,
  ConversationSummary,
  McpOverview,
  MemoryPayload,
  ModelsPayload,
  SettingsPayload,
  Status,
  StreamEvent,
} from "./types"

export const API_BASE =
  process.env.NEXT_PUBLIC_LOMA_API ?? "http://127.0.0.1:5050"

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init)
  if (!response.ok) {
    const detail = await response.text()
    throw new Error(detail || `${response.status} ${response.statusText}`)
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
    if (done) break
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
  tts_voice?: string
  temperature?: number
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
