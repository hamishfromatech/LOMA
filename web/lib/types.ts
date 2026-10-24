export type ToolCallView = {
  id: string
  name: string
  arguments: Record<string, unknown>
  result?: string
  error?: boolean
  latencyMs?: number
  status: "running" | "done" | "error"
}

export type TurnMetrics = {
  routeMs: number
  toolMs: number
  firstTokenS: number
  firstAudioS: number
  totalS: number
  contextTokens: number
}

export type ChatMessage = {
  id: string
  role: "user" | "assistant"
  content: string
  tools: ToolCallView[]
  intent?: string
  matched?: string
  metrics?: TurnMetrics
  streaming?: boolean
  error?: string
  memory?: string
}

export type Status = {
  ready: boolean
  models: {
    python_rss_mb: number
    ollama_mb: number
    total_mb: number
    models: { name: string; size_mb: number; runner: string }[]
  }
  memory_budget_mb: number
  context: {
    window: number
    policy: string
    live_tokens: number
    compact_threshold: number
  }
  tools: { name: string; description: string; required: string[] }[]
  config: {
    llm_model: string
    tts_model: string
    stt_model: string
    temperature: number
  }
}

export type StreamEvent =
  | { type: "route"; intent: string; matched: string; latency_ms: number }
  | {
      type: "tool"
      name: string
      arguments: Record<string, unknown>
      confidence: number
    }
  | {
      type: "tool_result"
      name: string
      result: string
      latency_ms: number
      error: boolean
    }
  | { type: "delta"; text: string }
  | { type: "audio"; sentence: string; b64: string; format: string }
  | {
      type: "done"
      reply: string
      intent: string
      matched: string
      tool: string[]
      tool_confidence: number
      timings: {
        route_ms: number
        tool_ms: number
        first_token_s: number
        first_audio_s: number
        total_s: number
      }
      context_tokens: number
      compaction: {
        messages_removed: number
        tokens_before: number
        tokens_after: number
      } | null
    }
  | MemoryEvent
  | { type: "error"; message: string }

export type ModelOption = {
  name: string
  size_mb: number
  family: string
  parameters: string
  capabilities: string[]
  /** Which Ollama listing produced it: api/tags, v1/models, or preference. */
  quantization?: string
  modified_at?: string
  source?: string
}

export type ModelsPayload = {
  models: ModelOption[]
  current: string
  source: string
  error: string | null
}

export type McpServer = {
  name: string
  transport: "stdio" | "http"
  command: string
  args: string[]
  url: string
  env: Record<string, string>
  connected: boolean
  tools: string[]
}

export type McpOverview = {
  path: string
  connected: boolean
  tool_count: number
  servers: McpServer[]
}

export type SettingsPayload = {
  llm_model: string
  tts_voice: string
  temperature: number
  options: {
    llm_models: ModelOption[]
    voices: string[]
  }
  mcp: McpOverview
  memory: MemoryPayload
  context: { window: number; policy: string }
}

export type ConversationSummary = {
  id: string
  title: string
  created_at: string
  updated_at: string
  message_count: number
}

export type Conversation = {
  id: string
  title: string
  created_at: string
  updated_at: string
  last_reply: string
  messages: { role: "user" | "assistant"; content: string }[]
}

export type MemoryPayload = {
  enabled: boolean
  auto_extract: boolean
  path: string
  content: string
  facts: string[]
}

export type MemoryEvent = { type: "memory"; fact: string; stored: boolean }
