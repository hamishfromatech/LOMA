"use client"

import { ChatsDrawer } from "@/components/sidebar"
import { IconRail } from "@/components/icon-rail"
import { MicButton } from "@/components/mic-button"
import { TurnMetrics } from "@/components/turn-metrics"
import { useAudioQueue } from "@/components/audio-queue"
import {
  ChatContainerContent,
  ChatContainerRoot,
  ChatContainerScrollAnchor,
} from "@/components/ui/chat-container"
import { DotsLoader } from "@/components/ui/loader"
import { Message, MessageContent } from "@/components/ui/message"
import {
  PromptInput,
  PromptInputActions,
  PromptInputTextarea,
} from "@/components/ui/prompt-input"
import {
  Reasoning,
  ReasoningContent,
  ReasoningTrigger,
} from "@/components/ui/reasoning"
import { ScrollButton } from "@/components/ui/scroll-button"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { SystemMessage } from "@/components/ui/system-message"
import { Tool, type ToolPart } from "@/components/ui/tool"
import { Button } from "@/components/ui/button"
import {
  createConversation,
  deleteConversation,
  getConversation,
  getSettings,
  getStatus,
  listConversations,
  streamChat,
  updateSettings,
} from "@/lib/api"
import type {
  ChatMessage,
  Conversation,
  ConversationSummary,
  SettingsPayload,
  Status,
  StreamEvent,
  ToolCallView,
} from "@/lib/types"
import {
  ArrowUp,
  AudioLines,
  Loader2,
  Settings2,
  Sparkle,
  Sparkles,
  Square,
  Volume2,
  VolumeX,
  Wrench,
  X,
} from "lucide-react"
import Link from "next/link"
import { useCallback, useEffect, useRef, useState } from "react"

const SUGGESTIONS = [
  { title: "Tell me a fun fact", subtitle: "about the Roman Empire" },
  { title: "Roll the dice", subtitle: "a twenty-sided die, for a game" },
  { title: "What's on this machine?", subtitle: "memory, CPU and OS" },
  { title: "Remember something", subtitle: "I prefer tea over coffee" },
]

function newId() {
  return Math.random().toString(36).slice(2, 10)
}

function toToolPart(tool: ToolCallView): ToolPart {
  return {
    type: tool.name,
    state:
      tool.status === "running"
        ? "input-available"
        : tool.status === "error"
          ? "output-error"
          : "output-available",
    input: tool.arguments,
    output: tool.result ? { result: tool.result } : undefined,
    toolCallId: tool.id,
    errorText: tool.error ? tool.result : undefined,
  }
}

function fromConversation(conversation: Conversation): ChatMessage[] {
  return conversation.messages.map((message, index) => ({
    id: `${conversation.id}-${index}-${newId()}`,
    role: message.role,
    content: message.content,
    tools: [],
  }))
}

// ---------------------------------------------------------------------------
// Composer — one large rounded surface, controls along the bottom
// ---------------------------------------------------------------------------

type ComposerProps = {
  value: string
  onValueChange: (value: string) => void
  onSubmit: () => void
  busy: boolean
  onStop: () => void
  voiceOn: boolean
  onToggleVoice: () => void
  onTranscript: (text: string) => void
  settings: SettingsPayload | null
  onModelChange: (model: string) => void
}

function Composer({
  value,
  onValueChange,
  onSubmit,
  busy,
  onStop,
  voiceOn,
  onToggleVoice,
  onTranscript,
  settings,
  onModelChange,
}: ComposerProps) {
  return (
    <PromptInput
      value={value}
      onValueChange={onValueChange}
      onSubmit={onSubmit}
      isLoading={busy}
      className="rounded-[26px] border border-white/[0.07] bg-[#1f1f1f] px-4 pb-2.5 pt-3.5 shadow-[0_1px_0_rgba(255,255,255,0.04)_inset,0_18px_40px_-24px_rgba(0,0,0,0.9)]"
    >
      <PromptInputTextarea
        id="loma-prompt"
        name="message"
        aria-label="Message LOMA"
        placeholder="How can I help you today?"
        className="min-h-[46px] rounded-none border-0 bg-transparent px-1 py-1 text-[15px] leading-relaxed shadow-none placeholder:text-white/35 focus-visible:border-0 focus-visible:ring-0 dark:bg-transparent"
      />

      <PromptInputActions className="justify-between px-0 pb-0 pt-1">
        <div className="flex items-center gap-1">
          <Button
            type="button"
            variant="ghost"
            size="icon-sm"
            onClick={onToggleVoice}
            aria-label={voiceOn ? "Mute spoken replies" : "Speak replies"}
            title={voiceOn ? "Spoken replies on" : "Spoken replies off"}
            className="rounded-full text-white/55 hover:bg-white/[0.07] hover:text-white"
          >
            {voiceOn ? (
              <Volume2 className="size-4" />
            ) : (
              <VolumeX className="size-4" />
            )}
          </Button>

          <span className="hidden items-center gap-1.5 px-1 font-mono text-[11px] text-white/25 sm:flex">
            <Wrench className="size-3.5" />
            offline
          </span>
        </div>

        <div className="flex items-center gap-1">
          {settings && (
            <Select
              value={settings.llm_model}
              onValueChange={(next) => onModelChange(String(next))}
            >
              <SelectTrigger
                size="sm"
                aria-label="Language model"
                className="h-8 gap-1 border-0 bg-transparent px-2 font-mono text-[12px] text-white/60 shadow-none hover:bg-white/[0.05] hover:text-white/90"
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {settings.options.llm_models.map((model) => (
                  <SelectItem key={model.name} value={model.name}>
                    {model.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}

          <MicButton onTranscript={onTranscript} disabled={busy} />

          <Button
            type="button"
            size="icon"
            onClick={
              busy
                ? onStop
                : value.trim()
                  ? onSubmit
                  : () => document.getElementById("loma-prompt")?.focus()
            }
            aria-label={busy ? "Stop" : value.trim() ? "Send" : "Start typing"}
            className="size-9 rounded-full bg-white text-black transition hover:bg-white/90"
          >
            {busy ? (
              <Square className="size-3.5" />
            ) : value.trim() ? (
              <ArrowUp className="size-4" />
            ) : (
              <AudioLines className="size-4" />
            )}
          </Button>
        </div>
      </PromptInputActions>
    </PromptInput>
  )
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

export function ChatView() {
  const [conversations, setConversations] = useState<ConversationSummary[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState("")
  const [busy, setBusy] = useState(false)
  const [voiceOn, setVoiceOn] = useState(true)
  const [status, setStatus] = useState<Status | null>(null)
  const [settings, setSettings] = useState<SettingsPayload | null>(null)
  const [chatsOpen, setChatsOpen] = useState(false)
  const [bannerOpen, setBannerOpen] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const abort = useRef<AbortController | null>(null)
  const { enqueue, unlock, stop } = useAudioQueue()

  const refreshStatus = useCallback(async () => {
    try {
      setStatus(await getStatus())
    } catch {
      setStatus(null)
    }
  }, [])

  const refreshConversations = useCallback(async () => {
    try {
      setConversations(await listConversations())
    } catch {
      /* the drawer is not worth an error banner */
    }
  }, [])

  useEffect(() => {
    void refreshStatus()
    void refreshConversations()
    void getSettings()
      .then(setSettings)
      .catch(() => undefined)
    const timer = window.setInterval(refreshStatus, 10_000)
    return () => window.clearInterval(timer)
  }, [refreshStatus, refreshConversations])

  const openConversation = useCallback(
    async (id: string) => {
      abort.current?.abort()
      stop()
      try {
        const conversation = await getConversation(id)
        setActiveId(conversation.id)
        setMessages(fromConversation(conversation))
        setChatsOpen(false)
        setError(null)
      } catch {
        setError("could not open that conversation")
      }
    },
    [stop],
  )

  const newChat = useCallback(() => {
    abort.current?.abort()
    stop()
    setActiveId(null)
    setMessages([])
    setInput("")
    setError(null)
    setChatsOpen(false)
  }, [stop])

  const removeChat = useCallback(
    async (id: string) => {
      try {
        await deleteConversation(id)
        setConversations((current) => current.filter((item) => item.id !== id))
        if (id === activeId) newChat()
      } catch {
        setError("could not delete that conversation")
      }
    },
    [activeId, newChat],
  )

  const changeModel = useCallback(
    async (model: string) => {
      try {
        setSettings(await updateSettings({ llm_model: model }))
        void refreshStatus()
      } catch {
        setError("could not switch the model")
      }
    },
    [refreshStatus],
  )

  const send = useCallback(
    async (text: string) => {
      const message = text.trim()
      if (!message || busy) return

      unlock()
      setError(null)
      setInput("")
      setBusy(true)
      setChatsOpen(false)

      let conversationId = activeId
      if (!conversationId) {
        try {
          const created = await createConversation()
          conversationId = created.id
          setActiveId(created.id)
        } catch {
          setError("could not start a conversation")
          setBusy(false)
          return
        }
      }

      const userMessage: ChatMessage = {
        id: newId(),
        role: "user",
        content: message,
        tools: [],
      }
      const assistantId = newId()
      const assistantMessage: ChatMessage = {
        id: assistantId,
        role: "assistant",
        content: "",
        tools: [],
        streaming: true,
      }
      setMessages((current) => [...current, userMessage, assistantMessage])

      const patch = (update: (item: ChatMessage) => ChatMessage) => {
        setMessages((current) =>
          current.map((item) => (item.id === assistantId ? update(item) : item)),
        )
      }

      const handle = (event: StreamEvent) => {
        switch (event.type) {
          case "route":
            patch((item) => ({
              ...item,
              intent: event.intent,
              matched: event.matched,
            }))
            break

          case "tool":
            patch((item) => ({
              ...item,
              tools: [
                ...item.tools,
                {
                  id: newId(),
                  name: event.name,
                  arguments: event.arguments,
                  status: "running",
                },
              ],
            }))
            break

          case "tool_result":
            patch((item) => {
              const tools = [...item.tools]
              for (let index = tools.length - 1; index >= 0; index -= 1) {
                if (
                  tools[index].name === event.name &&
                  tools[index].status === "running"
                ) {
                  tools[index] = {
                    ...tools[index],
                    result: event.result,
                    error: event.error,
                    latencyMs: event.latency_ms,
                    status: event.error ? "error" : "done",
                  }
                  break
                }
              }
              return { ...item, tools }
            })
            break

          case "delta":
            patch((item) => ({ ...item, content: item.content + event.text }))
            break

          case "audio":
            enqueue(event.b64)
            break

          case "memory":
            if (event.stored) patch((item) => ({ ...item, memory: event.fact }))
            break

          case "done":
            patch((item) => ({
              ...item,
              streaming: false,
              content: event.reply || item.content,
              metrics:
                event.timings && typeof event.timings.total_s === "number"
                  ? {
                      routeMs: event.timings.route_ms ?? 0,
                      toolMs: event.timings.tool_ms ?? 0,
                      firstTokenS: event.timings.first_token_s ?? 0,
                      firstAudioS: event.timings.first_audio_s ?? 0,
                      totalS: event.timings.total_s ?? 0,
                      contextTokens: event.context_tokens ?? 0,
                    }
                  : item.metrics,
            }))
            void refreshConversations()
            void refreshStatus()
            break

          case "error":
            patch((item) => ({ ...item, streaming: false, error: event.message }))
            setError(event.message)
            break
        }
      }

      const controller = new AbortController()
      abort.current = controller
      try {
        await streamChat(message, handle, {
          conversationId: conversationId!,
          speak: voiceOn,
          signal: controller.signal,
        })
      } catch (cause) {
        if ((cause as Error).name !== "AbortError") {
          const detail = cause instanceof Error ? cause.message : "request failed"
          setError(detail)
          patch((item) => ({ ...item, streaming: false, error: detail }))
        }
      } finally {
        abort.current = null
        setBusy(false)
        void refreshConversations()
      }
    },
    [activeId, busy, enqueue, refreshConversations, refreshStatus, unlock, voiceOn],
  )

  const stopTurn = useCallback(() => {
    abort.current?.abort()
    stop()
    setBusy(false)
  }, [stop])

  const empty = messages.length === 0
  const modelName = settings?.llm_model ?? "loading"

  return (
    <div className="flex h-dvh bg-[#171717] text-[#eeeeee]">
      <IconRail
        onNew={newChat}
        onToggleChats={() => setChatsOpen((open) => !open)}
        chatsOpen={chatsOpen}
      />

      <ChatsDrawer
        conversations={conversations}
        activeId={activeId}
        open={chatsOpen}
        onClose={() => setChatsOpen(false)}
        onNew={newChat}
        onSelect={(id) => void openConversation(id)}
        onDelete={(id) => void removeChat(id)}
      />

      <div className="relative flex min-w-0 flex-1 flex-col">
        {/* top-right utilities */}
        <div className="pointer-events-none absolute right-3 top-3 z-20 flex items-center gap-2">
          <span className="pointer-events-auto flex items-center gap-1.5 rounded-full px-2 py-1 font-mono text-[10px] text-white/35">
            {status?.ready ? (
              <span className="size-1.5 rounded-full bg-emerald-400" />
            ) : (
              <Loader2 className="size-3 animate-spin" />
            )}
            {status ? `${Math.round(status.models.total_mb)} MB` : "…"}
          </span>
          <Link
            href="/settings"
            aria-label="Settings"
            className="pointer-events-auto grid size-8 place-items-center rounded-lg text-white/40 transition hover:bg-white/[0.06] hover:text-white"
          >
            <Settings2 className="size-4" />
          </Link>
        </div>

        {bannerOpen && (
          <div className="z-10 flex items-center gap-2 border-b border-white/[0.06] bg-white/[0.02] py-1.5 pl-3 pr-24 text-[12px]">
            <span className="rounded bg-sky-500/15 px-1.5 py-0.5 font-mono text-[10px] font-medium uppercase tracking-wide text-sky-300">
              info
            </span>
            <span className="truncate text-white/60">
              LOMA runs entirely on this machine — every model, tool and memory
              file stays local.
            </span>
            <button
              onClick={() => setBannerOpen(false)}
              aria-label="Dismiss"
              className="ml-auto text-white/30 hover:text-white"
            >
              <X className="size-3.5" />
            </button>
          </div>
        )}

        {empty ? (
          <div className="flex min-h-0 flex-1 flex-col items-center justify-center overflow-y-auto px-4 pb-24">
            <div className="w-full max-w-[760px]">
              <div className="mb-7 flex items-center justify-center gap-3">
                <span className="grid size-9 place-items-center rounded-xl bg-white text-black">
                  <Sparkles className="size-4" />
                </span>
                <span className="text-[28px] font-medium tracking-tight text-white/95 sm:text-[32px]">
                  {modelName}
                </span>
              </div>

              <Composer
                value={input}
                onValueChange={setInput}
                onSubmit={() => void send(input)}
                busy={busy}
                onStop={stopTurn}
                voiceOn={voiceOn}
                onToggleVoice={() => setVoiceOn((value) => !value)}
                onTranscript={(text) => void send(text)}
                settings={settings}
                onModelChange={(model) => void changeModel(model)}
              />

              <div className="mt-6">
                <p className="mb-2 flex items-center gap-1.5 text-[13px] text-white/45">
                  <Sparkle className="size-3.5" />
                  Suggested
                </p>
                <div className="grid gap-x-10 gap-y-3 sm:grid-cols-2">
                  {SUGGESTIONS.map((suggestion) => (
                    <button
                      key={suggestion.title}
                      onClick={() =>
                        void send(`${suggestion.title} ${suggestion.subtitle}`)
                      }
                      className="rounded-lg py-1 text-left transition hover:opacity-80"
                    >
                      <p className="text-[14px] text-white/85">{suggestion.title}</p>
                      <p className="text-[13px] leading-snug text-white/40">
                        {suggestion.subtitle}
                      </p>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        ) : (
          <>
            <main className="relative min-h-0 flex-1">
              <ChatContainerRoot className="flex h-full flex-col overflow-y-auto">
                <ChatContainerContent className="mx-auto w-full max-w-[760px] space-y-7 px-4 py-8">
                  {messages.map((message) =>
                    message.role === "user" ? (
                      <Message key={message.id} className="justify-end">
                        <MessageContent className="max-w-[85%] rounded-2xl rounded-br-md bg-white/[0.08] px-4 py-2.5 text-[14px] text-white/90">
                          {message.content}
                        </MessageContent>
                      </Message>
                    ) : (
                      <Message key={message.id} className="gap-3">
                        <span className="mt-0.5 grid size-7 shrink-0 place-items-center rounded-lg bg-white text-black">
                          <Sparkles className="size-3.5" />
                        </span>

                        <div className="min-w-0 flex-1 space-y-3">
                          {message.tools.map((tool) => (
                            <Tool key={tool.id} toolPart={toToolPart(tool)} />
                          ))}

                          {message.content && (
                            <MessageContent
                              markdown
                              className="prose-invert max-w-none bg-transparent p-0 text-[14px] leading-relaxed"
                            >
                              {message.content}
                            </MessageContent>
                          )}

                          {message.streaming && !message.content && (
                            <div className="flex items-center gap-2 py-1">
                              <DotsLoader />
                              <span className="font-mono text-[10px] uppercase tracking-wider text-white/30">
                                thinking
                              </span>
                            </div>
                          )}

                          {message.memory && (
                            <SystemMessage variant="action" fill={false}>
                              Remembered: {message.memory}
                            </SystemMessage>
                          )}

                          {message.metrics && (
                            <>
                              {message.tools.length > 0 && (
                                <Reasoning>
                                  <ReasoningTrigger className="font-mono text-[10px] text-white/30 hover:text-white/60">
                                    {message.tools.length} tool call
                                    {message.tools.length > 1 ? "s" : ""} ·{" "}
                                    {message.metrics.totalS.toFixed(2)}s
                                  </ReasoningTrigger>
                                  <ReasoningContent className="mt-2 rounded-lg border border-white/5 bg-white/[0.02] p-3">
                                    <div className="space-y-1 font-mono text-[11px] text-white/45">
                                      {message.tools.map((tool) => (
                                        <p key={tool.id}>
                                          {tool.name}(
                                          {JSON.stringify(tool.arguments)}) →{" "}
                                          {tool.result?.slice(0, 140) ?? "…"}
                                        </p>
                                      ))}
                                      <p>
                                        route: {message.intent ?? "answer"}
                                        {message.matched ? ` · "${message.matched}"` : ""}
                                        {" · "}
                                        {message.metrics.routeMs.toFixed(2)} ms
                                      </p>
                                    </div>
                                  </ReasoningContent>
                                </Reasoning>
                              )}

                              <TurnMetrics
                                metrics={message.metrics}
                                intent={message.intent}
                                matched={message.matched}
                              />
                            </>
                          )}

                          {message.error && (
                            <p className="rounded-lg border border-red-500/30 bg-red-500/5 px-3 py-2 text-xs text-red-300">
                              {message.error}
                            </p>
                          )}
                        </div>
                      </Message>
                    ),
                  )}
                </ChatContainerContent>

                <ChatContainerScrollAnchor />

                <div className="pointer-events-none sticky bottom-4 z-10 -mt-14 flex w-full justify-center">
                  <ScrollButton className="pointer-events-auto" />
                </div>
              </ChatContainerRoot>
            </main>

            <footer className="bg-gradient-to-t from-[#171717] via-[#171717] to-transparent px-4 pb-4 pt-2">
              <div className="mx-auto w-full max-w-[760px]">
                <Composer
                  value={input}
                  onValueChange={setInput}
                  onSubmit={() => void send(input)}
                  busy={busy}
                  onStop={stopTurn}
                  voiceOn={voiceOn}
                  onToggleVoice={() => setVoiceOn((value) => !value)}
                  onTranscript={(text) => void send(text)}
                  settings={settings}
                  onModelChange={(model) => void changeModel(model)}
                />
                <div className="mt-2 flex items-center justify-between px-2">
                  <p className="font-mono text-[10px] text-white/20">
                    needle 3 · phonon · faster-whisper · offline
                  </p>
                  {error && (
                    <p className="truncate font-mono text-[10px] text-red-400">
                      {error}
                    </p>
                  )}
                </div>
              </div>
            </footer>
          </>
        )}

        {error && empty && (
          <p className="absolute bottom-4 left-1/2 -translate-x-1/2 font-mono text-[10px] text-red-400">
            {error}
          </p>
        )}
      </div>
    </div>
  )
}
