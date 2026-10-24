"use client"

import { AnswerMessage, UserMessage } from "@/components/answer"
import { LomaMark } from "@/components/brand"
import { Composer } from "@/components/composer"
import { Onboarding, type OnboardingAnswers } from "@/components/onboarding"
import { ChatsDrawer } from "@/components/sidebar"
import { IconRail } from "@/components/icon-rail"
import { StatusBar } from "@/components/status-bar"
import { useAudioQueue } from "@/components/audio-queue"
import {
  ChatContainerContent,
  ChatContainerRoot,
  ChatContainerScrollAnchor,
} from "@/components/ui/chat-container"
import { ScrollButton } from "@/components/ui/scroll-button"
import {
  completeOnboarding,
  createConversation,
  deleteConversation,
  getOnboarding,
  getConversation,
  getSettings,
  getStatus,
  listConversations,
  listModels,
  renameConversation,
  skipOnboarding,
  streamChat,
  updateSettings,
} from "@/lib/api"
import type {
  ChatMessage,
  Conversation,
  ConversationSummary,
  ModelOption,
  SettingsPayload,
  Status,
  StreamEvent,
} from "@/lib/types"
import { cn } from "@/lib/utils"
import {
  Brain,
  Dice5,
  FileText,
  Loader2,
  MonitorSmartphone,
  Sparkles,
} from "lucide-react"
import { useCallback, useEffect, useRef, useState } from "react"

const SUGGESTIONS = [
  {
    icon: MonitorSmartphone,
    title: "What's on this machine?",
    subtitle: "memory, CPU and OS",
    prompt: "What's on this machine?",
  },
  {
    icon: Brain,
    title: "Remember something",
    subtitle: "I prefer tea over coffee",
    prompt: "Remember that I prefer tea over coffee.",
  },
  {
    icon: FileText,
    title: "Summarise a file",
    subtitle: "paste it in, keep it local",
    prompt: "Summarise the file I paste next in three lines.",
  },
  {
    icon: Dice5,
    title: "Roll the dice",
    subtitle: "a twenty-sided die, for a game",
    prompt: "Roll a twenty-sided die for a game.",
  },
]

function newId() {
  return Math.random().toString(36).slice(2, 10)
}

function fromConversation(conversation: Conversation): ChatMessage[] {
  return conversation.messages.map((message, index) => ({
    id: `${conversation.id}-${index}-${newId()}`,
    role: message.role,
    content: message.content,
    tools: [],
  }))
}

function greeting() {
  const hour = new Date().getHours()
  if (hour < 5) return "Still up"
  if (hour < 12) return "Good morning"
  if (hour < 18) return "Good afternoon"
  return "Good evening"
}

/** Small always-visible health readout. Click for the full budget. */
function StatusPill({ status }: { status: Status | null }) {
  const [open, setOpen] = useState(false)

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        aria-label="Runtime status"
        className={cn(
          "flex h-7 items-center gap-2 rounded-full border px-2.5 font-mono text-[10.5px] transition",
          "border-loma-line bg-white/[0.03] hover:bg-white/[0.06]",
        )}
      >
        {status?.ready ? (
          <span className="size-1.5 rounded-full bg-loma-ok shadow-[0_0_6px] shadow-loma-ok/50" />
        ) : (
          <Loader2 className="size-3 animate-spin text-loma-faint" />
        )}
        <span className="tabular-nums text-loma-dim">
          {status ? `${(status.models.total_mb / 1024).toFixed(1)} GB` : "…"}
        </span>
        <span className="hidden text-loma-faint sm:inline">
          {status ? status.config.llm_model : "connecting"}
        </span>
      </button>

      {open && (
        <>
          <div
            className="fixed inset-0 z-40"
            onClick={() => setOpen(false)}
            aria-hidden
          />
          <div className="absolute right-0 top-9 z-50 w-[min(92vw,420px)] rounded-xl border border-loma-line bg-loma-raise p-3 shadow-2xl shadow-black/70">
            <p className="mb-2 font-mono text-[10px] uppercase tracking-[0.16em] text-loma-faint">
              runtime
            </p>
            <StatusBar status={status} />
            {status && (
              <p className="mt-3 border-t border-loma-line-soft pt-2 font-mono text-[10px] leading-relaxed text-loma-faint">
                {status.context.policy}
              </p>
            )}
          </div>
        </>
      )}
    </div>
  )
}

export function ChatView() {
  const [conversations, setConversations] = useState<ConversationSummary[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState("")
  const [busy, setBusy] = useState(false)
  const [voiceOn, setVoiceOn] = useState(true)
  const [status, setStatus] = useState<Status | null>(null)
  const [settings, setSettings] = useState<SettingsPayload | null>(null)
  const [models, setModels] = useState<ModelOption[]>([])
  const [attachments, setAttachments] = useState<{ id: string; dataUrl: string }[]>([])
  const [modelsLoading, setModelsLoading] = useState(false)
  const [chatsOpen, setChatsOpen] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [hello, setHello] = useState("Hello")
  const [onboarding, setOnboarding] = useState<{
    needed: boolean
    name: string
  } | null>(null)
  const [justOnboarded, setJustOnboarded] = useState(false)
  const abort = useRef<AbortController | null>(null)
  const { enqueue, unlock, stop } = useAudioQueue()

  useEffect(() => {
    setHello(greeting())
    void getOnboarding()
      .then((state) => setOnboarding({ needed: state.needed, name: state.name }))
      .catch(() => setOnboarding({ needed: false, name: "" }))
  }, [])

  const refreshStatus = useCallback(async () => {
    try {
      setStatus(await getStatus())
    } catch {
      setStatus(null)
    }
  }, [])

  const addAttachments = useCallback((files: File[]) => {
    const images = files.filter((file) => file.type.startsWith("image/")).slice(0, 4)
    for (const file of images) {
      const reader = new FileReader()
      reader.onload = () => {
        const dataUrl = String(reader.result ?? "")
        if (!dataUrl) return
        setAttachments((current) =>
          current.length >= 4
            ? current
            : [...current, { id: newId(), dataUrl }],
        )
      }
      reader.readAsDataURL(file)
    }
  }, [])

  const removeAttachment = useCallback((id: string) => {
    setAttachments((current) => current.filter((item) => item.id !== id))
  }, [])

  const refreshModels = useCallback(async () => {
    setModelsLoading(true)
    try {
      const payload = await listModels()
      setModels(payload.models)
    } catch {
      /* keep the last known list rather than blanking the picker */
    } finally {
      setModelsLoading(false)
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
    void refreshModels()
    void getSettings()
      .then((loaded) => {
        setSettings(loaded)
        // The stored preference wins over the optimistic default.
        setVoiceOn(loaded.speak_replies)
      })
      .catch(() => undefined)
    const timer = window.setInterval(refreshStatus, 10_000)
    return () => window.clearInterval(timer)
  }, [refreshStatus, refreshConversations, refreshModels])

  const finishOnboarding = useCallback(async (answers: OnboardingAnswers) => {
    const result = await completeOnboarding(answers)
    setOnboarding({ needed: false, name: result.summary.name || "" })
    setJustOnboarded(true)
    setSettings((current) =>
      current ? { ...current, memory: result.memory } : current,
    )
    void refreshConversations()
  }, [refreshConversations])

  const skipSetup = useCallback(async () => {
    await skipOnboarding()
    setOnboarding({ needed: false, name: "" })
  }, [])

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const meta = event.metaKey || event.ctrlKey
      if (meta && event.key.toLowerCase() === "k") {
        event.preventDefault()
        setChatsOpen((value) => !value)
      }
      if (event.key === "Escape") setChatsOpen(false)
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [])

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

  const renameChat = useCallback(
    async (id: string, title: string) => {
      setConversations((current) =>
        current.map((item) => (item.id === id ? { ...item, title } : item)),
      )
      try {
        await renameConversation(id, title)
      } catch {
        setError("could not rename that conversation")
      }
    },
    [],
  )

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

  const toggleVoice = useCallback(
    (next: boolean) => {
      setVoiceOn(next)
      void updateSettings({ speak_replies: next })
        .then(setSettings)
        .catch(() => undefined)
    },
    [],
  )

  const changeModel = useCallback(
    async (model: string) => {
      try {
        setSettings(await updateSettings({ llm_model: model }))
        setModels((current) => current.map((item) => item))
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
      const images = attachments.map((item) => item.dataUrl)
      if ((!message && images.length === 0) || busy) return

      unlock()
      setError(null)
      setInput("")
      setAttachments([])
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
        images,
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
              truncated: Boolean(event.truncated),
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
          images,
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
    [activeId, attachments, busy, enqueue, refreshConversations, refreshStatus, unlock, voiceOn],
  )

  const stopTurn = useCallback(() => {
    abort.current?.abort()
    stop()
    setBusy(false)
  }, [stop])

  const currentModel = models.find((item) => item.name === settings?.llm_model)
  const visionEnabled = currentModel?.capabilities.includes("vision") ?? false
  const empty = messages.length === 0
  const storedTitle = conversations.find((item) => item.id === activeId)?.title
  const firstLine = messages.find((item) => item.role === "user")?.content ?? ""
  const activeTitle =
    storedTitle && storedTitle !== "New chat"
      ? storedTitle
      : firstLine.trim().split("\n")[0].slice(0, 64) || "New chat"

  if (onboarding?.needed) {
    return (
      <Onboarding
        onComplete={finishOnboarding}
        onSkip={skipSetup}
      />
    )
  }

  return (
    <div className="flex h-dvh overflow-hidden bg-loma-bg text-loma-ink">
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
        onRename={(id, title) => void renameChat(id, title)}
      />

      <div className="relative flex min-w-0 flex-1 flex-col">
        {empty && (
          <div
            aria-hidden
            className="pointer-events-none absolute inset-x-0 top-0 h-[440px] bg-[radial-gradient(58%_100%_at_50%_0%,rgba(227,168,92,0.055),transparent_72%)]"
          />
        )}

        <header className="flex h-12 shrink-0 items-center gap-3 px-4 sm:px-6">
          <p className="min-w-0 flex-1 truncate text-[12.5px] text-loma-faint">
            {empty ? "" : activeTitle}
          </p>
          <StatusPill status={status} />
        </header>

        {empty ? (
          <div className="flex min-h-0 flex-1 flex-col items-center justify-center overflow-y-auto px-4 pb-16">
            <div className="w-full max-w-[720px]">
              <div className="mb-8 flex flex-col items-center text-center">
                <LomaMark size={44} className="mb-5" />
                <h1 className="text-[27px] font-medium tracking-[-0.02em] text-loma-ink sm:text-[31px]">
                  {onboarding?.name ? `${hello}, ${onboarding.name}.` : `${hello}.`}
                </h1>
                <p className="mt-2 max-w-[440px] text-[13.5px] leading-relaxed text-loma-faint">
                  {justOnboarded ? (
                    <>
                      I have written that to memory. Ask me anything — you can
                      change it later in{" "}
                      <a href="/settings" className="text-loma-dim underline decoration-loma-accent/40 underline-offset-2 hover:text-loma-ink">
                        Settings
                      </a>
                      .
                    </>
                  ) : (
                    <>
                      Ask anything. Every model, tool and memory file stays on this
                      machine — no network, no API bills.
                    </>
                  )}
                </p>
              </div>

              <Composer
                value={input}
                onValueChange={setInput}
                onSubmit={() => void send(input)}
                busy={busy}
                onStop={stopTurn}
                voiceOn={voiceOn}
                onToggleVoice={() => toggleVoice(!voiceOn)}
                onTranscript={(text) => void send(text)}
                settings={settings}
                models={models}
                onModelChange={(model) => void changeModel(model)}
                onRefreshModels={() => void refreshModels()}
                modelsLoading={modelsLoading}
                attachments={attachments}
                onAddAttachments={addAttachments}
                onRemoveAttachment={removeAttachment}
                visionEnabled={visionEnabled}
                variant="hero"
              />

              <div className="mt-8">
                <p className="mb-2.5 flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-[0.16em] text-loma-faint">
                  <Sparkles className="size-3" />
                  try
                </p>
                <div className="grid gap-2 sm:grid-cols-2">
                  {SUGGESTIONS.map((suggestion) => (
                    <button
                      key={suggestion.title}
                      onClick={() => void send(suggestion.prompt)}
                      className={cn(
                        "group flex items-start gap-2.5 rounded-xl border border-loma-line-soft bg-white/[0.015] px-3 py-2.5 text-left",
                        "transition hover:border-loma-line hover:bg-white/[0.04]",
                      )}
                    >
                      <suggestion.icon className="mt-0.5 size-3.5 shrink-0 text-loma-faint transition group-hover:text-loma-accent" />
                      <span className="min-w-0">
                        <span className="block truncate text-[13px] text-loma-ink/90">
                          {suggestion.title}
                        </span>
                        <span className="block truncate text-[11.5px] text-loma-faint">
                          {suggestion.subtitle}
                        </span>
                      </span>
                    </button>
                  ))}
                </div>
              </div>

              {error && (
                <p className="mt-6 text-center font-mono text-[10.5px] text-loma-danger">
                  {error}
                </p>
              )}
            </div>
          </div>
        ) : (
          <>
            <main className="relative min-h-0 flex-1">
              <ChatContainerRoot className="flex h-full flex-col overflow-y-auto">
                <ChatContainerContent className="mx-auto w-full max-w-[760px] space-y-8 px-4 pb-6 pt-2 sm:px-6">
                  {messages.map((message) =>
                    message.role === "user" ? (
                      <UserMessage key={message.id} message={message} />
                    ) : (
                      <AnswerMessage key={message.id} message={message} />
                    ),
                  )}
                </ChatContainerContent>

                <ChatContainerScrollAnchor />

                <div className="pointer-events-none sticky bottom-2 z-10 -mt-12 flex w-full justify-center">
                  <ScrollButton className="pointer-events-auto border border-loma-line bg-loma-raise text-loma-dim hover:text-loma-ink" />
                </div>
              </ChatContainerRoot>
            </main>

            <footer className="shrink-0 bg-gradient-to-t from-loma-bg via-loma-bg to-transparent px-4 pb-3 pt-2 sm:px-6">
              <div className="mx-auto w-full max-w-[760px]">
                <Composer
                  value={input}
                  onValueChange={setInput}
                  onSubmit={() => void send(input)}
                  busy={busy}
                  onStop={stopTurn}
                  voiceOn={voiceOn}
                  onToggleVoice={() => toggleVoice(!voiceOn)}
                  onTranscript={(text) => void send(text)}
                  settings={settings}
                  models={models}
                  onModelChange={(model) => void changeModel(model)}
                  onRefreshModels={() => void refreshModels()}
                  modelsLoading={modelsLoading}
                  attachments={attachments}
                  onAddAttachments={addAttachments}
                  onRemoveAttachment={removeAttachment}
                  visionEnabled={visionEnabled}
                />
                {error && (
                  <p className="mt-1 truncate px-1 text-center font-mono text-[10px] text-loma-danger">
                    {error}
                  </p>
                )}
              </div>
            </footer>
          </>
        )}
      </div>
    </div>
  )
}
