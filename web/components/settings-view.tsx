"use client"

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
import { useCallback, useEffect, useState } from "react"

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

  // add-server form
  const [name, setName] = useState("")
  const [transport, setTransport] = useState<"stdio" | "http">("stdio")
  const [command, setCommand] = useState("")
  const [args, setArgs] = useState("")
  const [url, setUrl] = useState("")
  const [env, setEnv] = useState("")

  // memory editor
  const [memoryDraft, setMemoryDraft] = useState("")

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
      document
        .querySelectorAll<HTMLInputElement>('input[id$="-hidden-input"]')
        .forEach((element) => {
          element.setAttribute("aria-hidden", "true")
          element.tabIndex = -1
        })
    }
    mark()
    const observer = new MutationObserver(mark)
    observer.observe(document.body, { childList: true, subtree: true })
    return () => observer.disconnect()
  }, [])

  const flashSaved = useCallback(() => {
    setSaved(true)
    window.setTimeout(() => setSaved(false), 1600)
  }, [])

  const change = useCallback(
    async (patch: { llm_model?: string; tts_voice?: string; temperature?: number }) => {
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
      const element = new Audio(URL.createObjectURL(blob))
      await element.play()
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
    setBusy(`remove-${server}`)
    setError(null)
    try {
      const overview = await removeMcpServer(server)
      setSettings((current) => (current ? { ...current, mcp: overview } : current))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "could not remove the server")
    } finally {
      setBusy(null)
    }
  }, [])

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
    setBusy("memory-clear")
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
  }, [])

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
    return (
      <div className="grid h-dvh place-items-center">
        <div className="flex items-center gap-3 text-sm text-loma-faint">
          <Loader2 className="size-4 animate-spin" />
          loading settings…
        </div>
      </div>
    )
  }

  const selectedModel = settings.options.llm_models.find(
    (model) => model.name === settings.llm_model,
  )

  return (
    <div className="mx-auto w-full max-w-3xl px-4 pb-16">
      <div className="sticky top-0 z-20 -mx-4 mb-6 border-b border-loma-line-soft bg-loma-bg/85 px-4 py-3 backdrop-blur-md">
        <div className="flex items-center gap-3">
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
          <LomaMark size={26} />
          <div className="leading-none">
            <h1 className="font-mono text-[11px] uppercase tracking-[0.2em] text-loma-ink">
              Settings
            </h1>
            <p className="mt-1.5 font-mono text-[9.5px] text-loma-faint">
              applied on the next turn, no restart
            </p>
          </div>
          <div
            className={cn(
              "ml-auto flex h-6 items-center gap-1.5 font-mono text-[10px] uppercase tracking-[0.12em] text-loma-ok",
              "transition-opacity duration-200",
              saved ? "opacity-100" : "opacity-0",
            )}
            aria-live="polite"
          >
            <Check className="size-3" /> saved
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
              <p className="font-mono text-[10px] text-loma-faint">
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
              onChange={(event) =>
                setSettings((current) =>
                  current
                    ? { ...current, temperature: Number(event.target.value) }
                    : current,
                )
              }
              onMouseUp={(event) =>
                void change({ temperature: Number(event.currentTarget.value) })
              }
              onTouchEnd={(event) =>
                void change({ temperature: Number(event.currentTarget.value) })
              }
              style={{ ["--fill" as string]: `${Number(settings.temperature) * 100}%` }}
              className="loma-range w-full"
            />
            <p className="text-[11px] leading-relaxed text-loma-faint">
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
          <p className="text-[11px] text-loma-faint">
            The list comes from the speech model itself. Phonon adds named voices
            as it ships them; this build offers {settings.options.voices.length}.
          </p>
        </Section>

        {/* ---------------------------------------------------------- */}
        <Section
          title="Long-term memory"
          description="A Markdown file LOMA reads at the start of every turn and appends to when it learns something durable."
        >
          <div className="flex flex-wrap items-center gap-2 text-xs text-loma-dim">
            <Brain className="size-4 text-loma-accent/80" />
            {plural(settings.memory.facts.length, "fact")} remembered
            <span className="font-mono text-[10px] text-loma-faint">
              {settings.memory.path}
            </span>
          </div>

          <textarea
            value={memoryDraft}
            onChange={(event) => setMemoryDraft(event.target.value)}
            rows={10}
            spellCheck={false}
            aria-label="Memory contents"
            className="w-full resize-y rounded-xl border border-loma-line bg-black/30 px-4 py-3 font-mono text-xs leading-relaxed text-loma-ink/85 outline-none placeholder:text-loma-faint/60 focus:border-loma-accent/40"
            placeholder="# LOMA memory&#10;&#10;- Prefers tea over coffee&#10;- Runs a 27B model at Q4 on 32 GB"
          />

          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="max-w-sm text-[11px] leading-relaxed text-loma-faint">
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
                  try {
                    await resetOnboarding()
                    router.push("/")
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
                className="text-loma-faint hover:text-red-400"
              >
                {busy === "memory-clear" ? (
                  <Loader2 className="size-3.5 animate-spin" />
                ) : (
                  <Trash2 className="size-3.5" />
                )}
                Clear
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
              <p className="rounded-lg border border-dashed border-loma-line px-4 py-6 text-center text-xs text-loma-faint">
                No servers configured yet.
              </p>
            )}

            {settings.mcp.servers.map((server: McpServer) => (
              <div
                key={server.name}
                className="rounded-xl border border-loma-line bg-black/20 px-4 py-3"
              >
                <div className="flex items-start gap-3">
                  <Server className="mt-0.5 size-4 shrink-0 text-loma-faint" />
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-mono text-sm text-loma-ink">
                        {server.name}
                      </span>
                      <Badge
                        variant="secondary"
                        className="bg-white/[0.05] text-[10px] font-normal text-loma-faint"
                      >
                        {server.transport}
                      </Badge>
                      <span className="flex items-center gap-1.5 font-mono text-[10px] text-loma-faint">
                        <StatusDot on={server.connected} />
                        {server.connected
                          ? plural(server.tools.length, "tool")
                          : "offline"}
                      </span>
                    </div>
                    <p className="mt-1 truncate font-mono text-[11px] text-loma-faint">
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
                    className="text-loma-faint hover:text-red-400"
                  >
                    {busy === `remove-${server.name}` ? (
                      <Loader2 className="size-4 animate-spin" />
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
                className="w-full rounded-lg border border-loma-line bg-transparent px-3 py-2 font-mono text-xs text-loma-ink/90 outline-none placeholder:text-loma-faint/60 focus:border-loma-accent/40"
              />
            </div>

            <div className="flex items-center justify-between gap-3">
              <p className="font-mono text-[10px] text-loma-faint/80">
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
            <p className="text-[11px] leading-relaxed text-loma-faint">
              LOMA restarts the tool router after every change and rolls the file
              back if the server does not come up. <code>{"${LOMA_ROOT}"}</code> and{" "}
              <code>{"${LOMA_PYTHON}"}</code> are expanded when the server starts.
            </p>
          </div>
        </Section>

        <p className="pb-6 text-center font-mono text-[10px] text-loma-faint/80">
          {settings.context.policy}
        </p>
      </div>
    </div>
  )
}
