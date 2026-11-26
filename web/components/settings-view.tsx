"use client"

import { SkeletonRows } from "@/components/ui/skeleton"

import { LomaMark } from "@/components/brand"
import { Badge } from "@/components/ui/badge"
import { Button, buttonVariants } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import {
  addMcpServer,
  clearMemory,
  resetOnboarding,
  getSettings,
  previewVoice,
  reloadMcp,
  removeMcpServer,
  updateSettings,
  writeMemory,
} from "@/lib/api"
import type { McpServer, SettingsPayload } from "@/lib/types"
import { cn } from "@/lib/utils"
import {
  ArrowLeft,
  Brain,
  Check,
  Loader2,
  Play,
  Plus,
  RefreshCw,
  Save,
  Server,
  Trash2,
  TriangleAlert,
  Wand2,
} from "lucide-react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { useCallback, useEffect, useRef, useState } from "react"

function plural(count: number, word: string) {
  return `${count} ${word}${count === 1 ? "" : "s"}`
}

function StatusDot({ on }: { on: boolean }) {
  return (
    <span
      className={cn(
        "inline-block size-2 rounded-full",
        on ? "bg-loma-ok shadow-[0_0_8px] shadow-loma-ok/50" : "bg-white/20",
      )}
    />
  )
}

function Section({
  title,
  description,
  children,
}: {
  title: string
  description: string
  children: React.ReactNode
}) {
  return (
    <Card className="border-loma-line bg-loma-surface">
      <CardHeader>
        <CardTitle className="text-[15px] tracking-[-0.01em] text-loma-ink">{title}</CardTitle>
        <CardDescription className="text-[12.5px] leading-relaxed text-loma-faint">
          {description}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">{children}</CardContent>
    </Card>
  )
}

export function SettingsView() {
  const router = useRouter()
  const [settings, setSettings] = useState<SettingsPayload | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [saved, setSaved] = useState(false)
  const [memoryConfirm, setMemoryConfirm] = useState(false)
  const [mcpRemoveConfirm, setMcpRemoveConfirm] = useState<string | null>(null)

  // add-server form
  const [name, setName] = useState("")
  const [transport, setTransport] = useState<"stdio" | "http">("stdio")
  const [command, setCommand] = useState("")
  const [args, setArgs] = useState("")
  const [url, setUrl] = useState("")
  const [env, setEnv] = useState("")

  // memory editor
  const [memoryDraft, setMemoryDraft] = useState("")
  const memoryDirty =
    settings !== null && memoryDraft !== settings.memory.content
  const rootRef = useRef<HTMLDivElement>(null)
  const temperatureSaveTimeout = useRef<number | null>(null)

  // Warn before navigating away with unsaved memory edits.
  useEffect(() => {
    if (!memoryDirty) return
    const onBeforeUnload = (event: BeforeUnloadEvent) => {
      event.preventDefault()
      event.returnValue = ""
    }
    window.addEventListener("beforeunload", onBeforeUnload)
    return () => window.removeEventListener("beforeunload", onBeforeUnload)
  }, [memoryDirty])

  // Reset the save debounce when the temperature is saved through the other
  // code path so a stale timer can't overwrite a newer value.
  useEffect(() => () => {
    if (temperatureSaveTimeout.current !== null) {
      window.clearTimeout(temperatureSaveTimeout.current)
      temperatureSaveTimeout.current = null
    }
  }, [])

  // Auto-clear the "confirm?" state after 3s so the user doesn't have to
  // rush-click the button.
  useEffect(() => {
    if (!memoryConfirm) return
    const timer = window.setTimeout(() => setMemoryConfirm(false), 3000)
    return () => window.clearTimeout(timer)
  }, [memoryConfirm])
  useEffect(() => {
    if (!mcpRemoveConfirm) return
    const timer = window.setTimeout(() => setMcpRemoveConfirm(null), 3000)
    return () => window.clearTimeout(timer)
  }, [mcpRemoveConfirm])

  const load = useCallback(async () => {
    try {
      const loaded = await getSettings()
      setSettings(loaded)
      setMemoryDraft(loaded.memory.content)
      setError(null)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "could not load settings")
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  // Base UI's Select renders a hidden text input for form submission. It has no
  // label of its own — the combobox button is the real control — so hide the
  // proxy from assistive technology instead of leaving a phantom field.
  useEffect(() => {
    const mark = () => {
      rootRef.current
        ?.querySelectorAll<HTMLInputElement>('input[id$="-hidden-input"]')
        .forEach((element) => {
          element.setAttribute("aria-hidden", "true")
          element.tabIndex = -1
        })
    }
    mark()
    const observer = new MutationObserver(mark)
    if (rootRef.current) observer.observe(rootRef.current, { childList: true, subtree: true })
    return () => observer.disconnect()
  }, [settings])

  const flashSaved = useCallback(() => {
    setSaved(true)
    window.setTimeout(() => setSaved(false), 1600)
  }, [])

  const change = useCallback(
    async (patch: {
      llm_model?: string
      vlm_model?: string
      tts_voice?: string
      temperature?: number
      speak_replies?: boolean
    }) => {
      setBusy(Object.keys(patch)[0] ?? "save")
      setError(null)
      try {
        setSettings(await updateSettings(patch))
        flashSaved()
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : "could not save")
      } finally {
        setBusy(null)
      }
    },
    [flashSaved],
  )

  const preview = useCallback(async (voice: string) => {
    setBusy(`preview-${voice}`)
    setError(null)
    try {
      const blob = await previewVoice(voice)
      const url = URL.createObjectURL(blob)
      const element = new Audio(url)
      const done = () => {
        URL.revokeObjectURL(url)
      }
      element.onended = done
      element.onerror = done
      await element.play().catch(done)
    } catch {
      setError("could not play the preview")
    } finally {
      setBusy(null)
    }
  }, [])

  const addServer = useCallback(async () => {
    setBusy("add-server")
    setError(null)
    try {
      const overview = await addMcpServer({
        name,
        command: transport === "stdio" ? command : undefined,
        args: transport === "stdio" ? args : undefined,
        url: transport === "http" ? url : undefined,
        env: env.trim() || undefined,
      })
      setSettings((current) => (current ? { ...current, mcp: overview } : current))
      setName("")
      setCommand("")
      setArgs("")
      setUrl("")
      setEnv("")
      flashSaved()
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "could not add the server")
    } finally {
      setBusy(null)
    }
  }, [args, command, env, flashSaved, name, transport, url])

  const removeServer = useCallback(async (server: string) => {
    if (mcpRemoveConfirm === server) {
      setBusy(`remove-${server}`)
      setMcpRemoveConfirm(null)
      setError(null)
      try {
        const overview = await removeMcpServer(server)
        setSettings((current) => (current ? { ...current, mcp: overview } : current))
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : "could not remove the server")
      } finally {
        setBusy(null)
      }
    } else {
      setMcpRemoveConfirm(server)
    }
  }, [mcpRemoveConfirm])

  const saveMemory = useCallback(async () => {
    setBusy("memory")
    setError(null)
    try {
      const memory = await writeMemory(memoryDraft)
      setSettings((current) => (current ? { ...current, memory } : current))
      setMemoryDraft(memory.content)
      flashSaved()
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "could not save memory")
    } finally {
      setBusy(null)
    }
  }, [flashSaved, memoryDraft])

  const wipeMemory = useCallback(async () => {
    if (!memoryConfirm) {
      setMemoryConfirm(true)
      return
    }
    setBusy("memory-clear")
    setMemoryConfirm(false)
    setError(null)
    try {
      const memory = await clearMemory()
      setSettings((current) => (current ? { ...current, memory } : current))
      setMemoryDraft(memory.content)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "could not clear memory")
    } finally {
      setBusy(null)
    }
  }, [memoryConfirm])

  const reload = useCallback(async () => {
    setBusy("reload")
    try {
      const overview = await reloadMcp()
      setSettings((current) => (current ? { ...current, mcp: overview } : current))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "reload failed")
    } finally {
      setBusy(null)
    }
  }, [])

  if (!settings) {
    if (error) {
      return (
        <div className="grid h-dvh place-items-center">
          <div className="text-center">
            <p className="mb-4 text-sm text-loma-danger" role="alert">{error}</p>
            <button
              type="button"
              className={cn(
                buttonVariants({ variant: "ghost" }),
                "text-sm text-loma-faint",
              )}
              onClick={() => void load()}
            >
              retry
            </button>
          </div>
        </div>
      )
    }
    return (
      <div className="grid h-dvh place-items-center p-6" role="status" aria-label="Loading settings">
        <div className="w-full max-w-[560px]">
          <SkeletonRows count={4} className="space-y-3" />
          <SkeletonRows count={2} className="mt-6 space-y-3" />
        </div>
      </div>
    )
  }

  const selectedModel = settings.options.llm_models.find(
    (model) => model.name === settings.llm_model,
  )
  // A model with no reported capabilities is kept: "unknown" is not "cannot
  // see". The API is the authority on whether the configured one works.
  const visionModels = settings.options.llm_models.filter(
    (model) =>
      model.capabilities.length === 0 || model.capabilities.includes("vision"),
  )
  const selectedVisionModel = settings.options.llm_models.find(
    (model) => model.name === settings.vlm_model,
  )

  return (
    <main id="loma-main" ref={rootRef} className="mx-auto w-full max-w-3xl px-4 pb-16">
      <div className="sticky top-0 z-20 -mx-4 mb-6 border-b border-loma-line-soft bg-loma-bg/85 px-4 py-3 backdrop-blur-md">
        <div className="flex items-center gap-3">
          {memoryDirty ? (
            <button
              type="button"
              onClick={() => {
                if (window.confirm("Discard unsaved memory changes and leave?")) {
                  router.push("/")
                }
              }}
              aria-label="Back to chat"
              className={cn(
                buttonVariants({ variant: "ghost", size: "icon" }),
                "size-8 text-loma-faint hover:bg-white/[0.06] hover:text-loma-ink",
              )}
            >
              <ArrowLeft className="size-4" />
            </button>
          ) : (
            <Link
              href="/"
              aria-label="Back to chat"
              className={cn(
                buttonVariants({ variant: "ghost", size: "icon" }),
                "size-8 text-loma-faint hover:bg-white/[0.06] hover:text-loma-ink",
              )}
            >
              <ArrowLeft className="size-4" />
            </Link>
          )}
          <LomaMark size={26} />
          <div className="leading-none">
            <h1 className="font-mono text-[11px] uppercase tracking-[0.2em] text-loma-ink">
              Settings
            </h1>
            <p className="mt-1.5 font-mono text-[9.5px] text-loma-dim">
              applied on the next turn, no restart
            </p>
          </div>
          <div
            className={cn(
              "ml-auto flex h-6 items-center gap-1.5 font-mono text-[10px] uppercase tracking-[0.12em] text-loma-ok",
              saved ? "opacity-100" : "opacity-0",
            )}
            aria-live="assertive"
          >
            {saved && (
              <>
                <Check className="size-3" /> saved
              </>
            )}
          </div>
        </div>
      </div>

      {error && (
        <div className="mb-6 flex items-start gap-2 rounded-xl border border-loma-danger/25 bg-loma-danger/[0.06] px-4 py-3 text-[12.5px] text-loma-danger">
          <TriangleAlert className="mt-0.5 size-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      <div className="space-y-6">
        {/* ---------------------------------------------------------- */}
        <Section
          title="Language model"
          description="The model that writes the answers. Smaller is faster and lighter."
        >
          <div className="grid gap-2">
            <Label htmlFor="llm">Model</Label>
            <Select
              value={settings.llm_model}
              onValueChange={(value) => void change({ llm_model: String(value) })}
              disabled={busy !== null}
            >
              <SelectTrigger id="llm" aria-label="Language model" className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent
                alignItemWithTrigger={false}
                sideOffset={6}
                className="border border-loma-line p-1 shadow-2xl shadow-black/70"
              >
                {settings.options.llm_models.map((model) => (
                  <SelectItem key={model.name} value={model.name} className="pl-2.5">
                    {model.name} · {Math.round(model.size_mb)} MB
                    {model.parameters ? ` · ${model.parameters}` : ""}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {selectedModel && (
              <p className="font-mono text-[10px] text-loma-dim">
                {[
                  selectedModel.parameters,
                  selectedModel.family,
                  selectedModel.capabilities.join(", "),
                ]
                  .filter(Boolean)
                  .join(" · ")}
              </p>
            )}
          </div>

          <Separator className="bg-white/[0.05]" />

          <div className="grid gap-2">
            <Label htmlFor="vlm">Vision model</Label>
            <Select
              value={settings.vlm_model}
              onValueChange={(value) => void change({ vlm_model: String(value) })}
              disabled={busy !== null}
            >
              <SelectTrigger id="vlm" aria-label="Vision model" className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent
                alignItemWithTrigger={false}
                sideOffset={6}
                className="border border-loma-line p-1 shadow-2xl shadow-black/70"
              >
                {visionModels.map((model) => (
                  <SelectItem key={model.name} value={model.name} className="pl-2.5">
                    {model.name} · {Math.round(model.size_mb)} MB
                    {model.parameters ? ` · ${model.parameters}` : ""}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-[11.5px] leading-relaxed text-loma-dim">
              {settings.dynamic_models
                ? "An image turn switches to this model and unloads the language model first, so only one chat model sits in memory. It swaps back on the next text turn."
                : "Dynamic model loading is off, so the language model answers everything. Turn it on with LOMA_DYNAMIC_MODELS=1 to swap this model in for images."}
            </p>
            {selectedVisionModel && (
              <p className="font-mono text-[10px] text-loma-faint">
                {[
                  selectedVisionModel.parameters,
                  selectedVisionModel.family,
                  selectedVisionModel.capabilities.join(", "),
                ]
                  .filter(Boolean)
                  .join(" · ")}
              </p>
            )}
          </div>

          <Separator className="bg-white/[0.05]" />

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label htmlFor="temperature">Temperature</Label>
              <span className="font-mono text-xs text-loma-dim">
                {settings.temperature.toFixed(1)}
              </span>
            </div>
            <input
              id="temperature"
              aria-label="Temperature"
              type="range"
              min={0}
              max={1}
              step={0.1}
              value={settings.temperature}
              onChange={(event) => {
                // Update the number live, but don't round-trip on every tick —
                // the POST is debounced so a full drag is one request.
                const next = Number(event.target.value)
                setSettings((current) =>
                  current ? { ...current, temperature: next } : current,
                )
                if (temperatureSaveTimeout.current !== null) {
                  window.clearTimeout(temperatureSaveTimeout.current)
                }
                temperatureSaveTimeout.current = window.setTimeout(
                  () => void change({ temperature: next }),
                  350,
                )
              }}
              style={{ ["--fill" as string]: `${Number(settings.temperature) * 100}%` }}
              className="loma-range w-full"
            />
            <p className="text-[11px] leading-relaxed text-loma-dim">
              Lower is more literal and repeats itself less. A small model starts
              narrating its own reasoning above about 0.3, and those words get
              spoken, so 0.2 is the default. Raise it if answers feel robotic.
            </p>
          </div>
        </Section>

        {/* ---------------------------------------------------------- */}
        <Section
          title="Voice"
          description="The Phonon voice used for spoken replies."
        >
          <div className="flex flex-wrap items-end gap-3">
            <div className="grid min-w-[200px] flex-1 gap-2">
              <Label htmlFor="voice">Voice</Label>
              <Select
                value={settings.tts_voice}
                onValueChange={(value) => void change({ tts_voice: String(value) })}
                disabled={busy !== null}
              >
                <SelectTrigger id="voice" aria-label="Voice" className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent
                  alignItemWithTrigger={false}
                  sideOffset={6}
                  className="border border-loma-line p-1 shadow-2xl shadow-black/70"
                >
                  {settings.options.voices.length === 0 && (
                    <div className="px-3 py-3 text-[11.5px] text-loma-faint">
                      no voices reported by the speech model
                    </div>
                  )}
                  {settings.options.voices.map((voice) => (
                    <SelectItem key={voice} value={voice} className="pl-2.5">
                      {voice}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <Button
              variant="outline"
              onClick={() => void preview(settings.tts_voice)}
              disabled={busy !== null}
              className="border-loma-line"
            >
              {busy === `preview-${settings.tts_voice}` ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <Play className="size-4" />
              )}
              Preview
            </Button>
          </div>
          <p className="text-[11px] text-loma-dim">
            The list comes from the speech model itself. Phonon adds named voices
            as it ships them; this build offers {settings.options.voices.length}.
          </p>
        </Section>

        {/* ---------------------------------------------------------- */}
        <Section
          title="Spoken replies"
          description="Whether LOMA reads its answers out loud. The same switch sits in the composer, next to the microphone."
        >
          <div className="flex items-start justify-between gap-4">
            <div className="min-w-0">
              <p className="text-[13px] text-loma-ink">
                {settings.speak_replies ? "Speaking replies" : "Text only"}
              </p>
              <p className="mt-1 max-w-md text-[11.5px] leading-relaxed text-loma-dim">
                {settings.speak_replies
                  ? "Each sentence is sent to the voice model as it is written, so playback starts before the answer finishes."
                  : "Answers arrive as text only. Nothing is sent to the speech model, which keeps the turn a little faster and quieter."}
              </p>
            </div>
            <button
              type="button"
              role="switch"
              aria-checked={settings.speak_replies}
              aria-label="Speak replies aloud"
              onClick={() => void change({ speak_replies: !settings.speak_replies })}
              disabled={busy !== null}
              className={cn(
                "relative h-6 w-11 shrink-0 rounded-full border transition-colors",
                settings.speak_replies
                  ? "border-loma-accent/40 bg-loma-accent/80"
                  : "border-loma-line bg-white/[0.08]",
              )}
            >
              <span
                className={cn(
                  "absolute top-[2px] size-[18px] rounded-full bg-[#14120f] transition-all duration-150",
                  settings.speak_replies ? "left-[22px]" : "left-[2px]",
                )}
              />
            </button>
          </div>
        </Section>

        {/* ---------------------------------------------------------- */}
        <Section
          title="Long-term memory"
          description="A Markdown file LOMA reads at the start of every turn and appends to when it learns something durable."
        >
          <div className="flex flex-wrap items-center gap-2 text-xs text-loma-dim">
            <Brain className="size-4 text-loma-accent/80" />
            {plural(settings.memory.facts.length, "fact")} remembered
            <span className="font-mono text-[10px] text-loma-dim">
              {settings.memory.path}
            </span>
          </div>

          <textarea
            value={memoryDraft}
            onChange={(event) => setMemoryDraft(event.target.value)}
            rows={10}
            spellCheck={false}
            aria-label="Memory contents"
            className="w-full resize-y rounded-xl border border-loma-line bg-black/30 px-4 py-3 font-mono text-xs leading-relaxed text-loma-ink/85 outline-none placeholder:text-loma-faint focus:border-loma-accent/40"
            placeholder="# LOMA memory&#10;&#10;- Prefers tea over coffee&#10;- Runs a 27B model at Q4 on 32 GB"
          />

          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="max-w-sm text-[11px] leading-relaxed text-loma-dim">
              Edit it by hand, delete a line, or say{" "}
              <span className="text-loma-dim">“remember that …”</span> in chat.
              Auto-extraction is {settings.memory.auto_extract ? "on" : "off"}.
            </p>
            <div className="flex gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={async () => {
                  setBusy("onboarding")
                  setError(null)
                  try {
                    await resetOnboarding()
                    router.push("/")
                  } catch (cause) {
                    setError(cause instanceof Error ? cause.message : "could not reset setup")
                  } finally {
                    setBusy(null)
                  }
                }}
                disabled={busy !== null}
                className="text-loma-faint hover:text-loma-ink"
                title="Walk through the setup questions again"
              >
                {busy === "onboarding" ? (
                  <Loader2 className="size-3.5 animate-spin" />
                ) : (
                  <Wand2 className="size-3.5" />
                )}
                Setup again
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => void wipeMemory()}
                disabled={busy !== null}
                className={cn(
                  "text-loma-faint hover:text-red-400",
                  memoryConfirm ? "bg-red-500/10" : "",
                )}
              >
                {busy === "memory-clear" ? (
                  <Loader2 className="size-3.5 animate-spin" />
                ) : memoryConfirm ? (
                  <Check className="size-3.5" />
                ) : (
                  <Trash2 className="size-3.5" />
                )}
                {memoryConfirm ? "confirm?" : "Clear"}
              </Button>
              <Button
                size="sm"
                onClick={() => void saveMemory()}
                disabled={busy !== null}
                className="bg-loma-accent text-[#1b1713] hover:bg-loma-accent/90"
              >
                {busy === "memory" ? (
                  <Loader2 className="size-3.5 animate-spin" />
                ) : (
                  <Save className="size-3.5" />
                )}
                Save memory
              </Button>
            </div>
          </div>
        </Section>

        {/* ---------------------------------------------------------- */}
        <Section
          title="MCP servers"
          description="Tools LOMA can call. Needle picks one and fills the arguments."
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs text-loma-dim">
              <StatusDot on={settings.mcp.connected} />
              {settings.mcp.connected
                ? `${plural(settings.mcp.tool_count, "tool")} across ${plural(settings.mcp.servers.length, "server")}`
                : "not connected"}
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => void reload()}
              disabled={busy !== null}
            >
              {busy === "reload" ? (
                <Loader2 className="size-3.5 animate-spin" />
              ) : (
                <RefreshCw className="size-3.5" />
              )}
              Reconnect
            </Button>
          </div>

          <div className="space-y-2">
            {settings.mcp.servers.length === 0 && (
              <p className="rounded-lg border border-dashed border-loma-line px-4 py-6 text-center text-xs text-loma-dim">
                No servers configured yet.
              </p>
            )}

            {settings.mcp.servers.map((server: McpServer) => (
              <div
                key={server.name}
                className="rounded-xl border border-loma-line bg-black/20 px-4 py-3"
              >
                <div className="flex items-start gap-3">
                  <Server className="mt-0.5 size-4 shrink-0 text-loma-dim" />
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-mono text-sm text-loma-ink">
                        {server.name}
                      </span>
                      <Badge
                        variant="secondary"
                        className="bg-white/[0.05] text-[10px] font-normal text-loma-dim"
                      >
                        {server.transport}
                      </Badge>
                      <span className="flex items-center gap-1.5 font-mono text-[10px] text-loma-dim">
                        <StatusDot on={server.connected} />
                        {server.connected
                          ? plural(server.tools.length, "tool")
                          : "offline"}
                      </span>
                    </div>
                    <p className="mt-1 truncate font-mono text-[11px] text-loma-dim">
                      {server.transport === "http"
                        ? server.url
                        : [server.command, ...server.args].join(" ")}
                    </p>
                    {server.tools.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-1.5">
                        {server.tools.map((tool) => (
                          <span
                            key={tool}
                            className="rounded bg-white/[0.05] px-1.5 py-0.5 font-mono text-[10px] text-loma-dim"
                          >
                            {tool}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    aria-label={`Remove ${server.name}`}
                    onClick={() => void removeServer(server.name)}
                    disabled={busy !== null}
                    className={cn(
                      "text-loma-faint hover:text-red-400",
                      mcpRemoveConfirm === server.name ? "bg-red-500/10" : "",
                    )}
                  >
                    {busy === `remove-${server.name}` ? (
                      <Loader2 className="size-4 animate-spin" />
                    ) : mcpRemoveConfirm === server.name ? (
                      <Check className="size-4" />
                    ) : (
                      <Trash2 className="size-4" />
                    )}
                  </Button>
                </div>
              </div>
            ))}
          </div>

          <Separator className="bg-white/[0.05]" />

          <div className="space-y-3">
            <p className="flex items-center gap-2 text-xs font-medium text-loma-dim">
              <Plus className="size-3.5" /> Add a server
            </p>

            <div className="grid gap-3 sm:grid-cols-2">
              <div className="grid gap-1.5">
                <Label htmlFor="server-name" className="text-[11px] text-loma-dim">
                  Name
                </Label>
                <Input
                  id="server-name"
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  placeholder="filesystem"
                  className="font-mono text-xs"
                />
              </div>

              <div className="grid gap-1.5">
                <Label className="text-[11px] text-loma-dim">Transport</Label>
                <div className="flex gap-1 rounded-lg border border-loma-line p-1">
                  {(["stdio", "http"] as const).map((kind) => (
                    <button
                      key={kind}
                      onClick={() => setTransport(kind)}
                      className={cn(
                        "flex-1 rounded-md px-2 py-1 font-mono text-[11px] transition",
                        transport === kind
                          ? "bg-white/[0.12] text-loma-ink"
                          : "text-loma-faint hover:text-loma-dim",
                      )}
                    >
                      {kind}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {transport === "stdio" ? (
              <div className="grid gap-3 sm:grid-cols-2">
                <div className="grid gap-1.5">
                  <Label htmlFor="server-command" className="text-[11px] text-loma-dim">
                    Command
                  </Label>
                  <Input
                    id="server-command"
                    value={command}
                    onChange={(event) => setCommand(event.target.value)}
                    placeholder="npx"
                    className="font-mono text-xs"
                  />
                </div>
                <div className="grid gap-1.5">
                  <Label htmlFor="server-args" className="text-[11px] text-loma-dim">
                    Arguments
                  </Label>
                  <Input
                    id="server-args"
                    value={args}
                    onChange={(event) => setArgs(event.target.value)}
                    placeholder="-y @modelcontextprotocol/server-filesystem /tmp"
                    className="font-mono text-xs"
                  />
                </div>
              </div>
            ) : (
              <div className="grid gap-1.5">
                <Label htmlFor="server-url" className="text-[11px] text-loma-dim">
                  URL
                </Label>
                <Input
                  id="server-url"
                  value={url}
                  onChange={(event) => setUrl(event.target.value)}
                  placeholder="https://example.com/mcp"
                  className="font-mono text-xs"
                />
              </div>
            )}

            <div className="grid gap-1.5">
              <Label htmlFor="server-env" className="text-[11px] text-loma-dim">
                Environment (optional JSON)
              </Label>
              <textarea
                id="server-env"
                aria-label="Server environment variables as JSON"
                value={env}
                onChange={(event) => setEnv(event.target.value)}
                rows={2}
                placeholder={'{"API_KEY": "..."}'}
                className="w-full rounded-lg border border-loma-line bg-transparent px-3 py-2 font-mono text-xs text-loma-ink/90 outline-none placeholder:text-loma-faint focus:border-loma-accent/40"
              />
            </div>

            <div className="flex items-center justify-between gap-3">
              <p className="font-mono text-[10px] text-loma-faint">
                saved to {settings.mcp.path}
              </p>
              <Button
                onClick={() => void addServer()}
                disabled={busy !== null || !name.trim() || (!command.trim() && !url.trim())}
                className="bg-loma-accent text-[#1b1713] hover:bg-loma-accent/90"
              >
                {busy === "add-server" ? (
                  <Loader2 className="size-4 animate-spin" />
                ) : (
                  <Plus className="size-4" />
                )}
                Add server
              </Button>
            </div>
            <p className="text-[11px] leading-relaxed text-loma-dim">
              LOMA restarts the tool router after every change and rolls the file
              back if the server does not come up. <code>{"${LOMA_ROOT}"}</code> and{" "}
              <code>{"${LOMA_PYTHON}"}</code> are expanded when the server starts.
            </p>
          </div>
        </Section>

        <p className="pb-6 text-center font-mono text-[10px] text-loma-dim">
          {settings.context.policy}
        </p>
      </div>
    </main>
  )
}
