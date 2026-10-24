"use client"

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
  Waves,
} from "lucide-react"
import Link from "next/link"
import { useCallback, useEffect, useState } from "react"

function StatusDot({ on }: { on: boolean }) {
  return (
    <span
      className={cn(
        "inline-block size-2 rounded-full",
        on ? "bg-emerald-400 shadow-[0_0_8px] shadow-emerald-400/50" : "bg-white/20",
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
    <Card className="border-white/10 bg-white/[0.03] backdrop-blur">
      <CardHeader>
        <CardTitle className="text-base">{title}</CardTitle>
        <CardDescription className="text-white/45">{description}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">{children}</CardContent>
    </Card>
  )
}

export function SettingsView() {
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
        <div className="flex items-center gap-3 text-sm text-white/40">
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
    <div className="mx-auto w-full max-w-3xl px-4 py-8">
      {/* ------------------------------------------------------------ */}
      <div className="mb-8 flex items-center gap-3">
        <Link
          href="/"
          aria-label="Back to chat"
          className={buttonVariants({ variant: "ghost", size: "icon" })}
        >
          <ArrowLeft className="size-4" />
        </Link>
        <div className="flex items-center gap-2.5">
          <span className="grid size-9 place-items-center rounded-xl bg-gradient-to-br from-amber-300 to-orange-500 text-black">
            <Waves className="size-5" />
          </span>
          <div className="leading-none">
            <h1 className="text-sm font-semibold tracking-[0.18em]">SETTINGS</h1>
            <p className="mt-1 font-mono text-[10px] text-white/40">
              applied on the next turn, no restart
            </p>
          </div>
        </div>
        <div className="ml-auto flex h-6 items-center gap-2 text-xs text-emerald-400">
          {saved && (
            <>
              <Check className="size-3.5" /> saved
            </>
          )}
        </div>
      </div>

      {error && (
        <div className="mb-6 flex items-start gap-2 rounded-xl border border-red-500/30 bg-red-500/5 px-4 py-3 text-xs text-red-300">
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
              <SelectContent>
                {settings.options.llm_models.map((model) => (
                  <SelectItem key={model.name} value={model.name}>
                    {model.name} · {Math.round(model.size_mb)} MB
                    {model.parameters ? ` · ${model.parameters}` : ""}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {selectedModel && (
              <p className="font-mono text-[10px] text-white/30">
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

          <Separator className="bg-white/5" />

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label htmlFor="temperature">Temperature</Label>
              <span className="font-mono text-xs text-white/50">
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
              className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-white/10 accent-amber-300"
            />
            <p className="text-[11px] leading-relaxed text-white/35">
              Lower is more literal and repeats itself less. 0.3–0.5 is the
              sweet spot for a spoken assistant.
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
                <SelectContent>
                  {settings.options.voices.map((voice) => (
                    <SelectItem key={voice} value={voice}>
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
              className="border-white/10"
            >
              {busy === `preview-${settings.tts_voice}` ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <Play className="size-4" />
              )}
              Preview
            </Button>
          </div>
          <p className="text-[11px] text-white/35">
            The list comes from the speech model itself. Phonon adds named voices
            as it ships them; this build offers {settings.options.voices.length}.
          </p>
        </Section>

        {/* ---------------------------------------------------------- */}
        <Section
          title="Long-term memory"
          description="A Markdown file LOMA reads at the start of every turn and appends to when it learns something durable."
        >
          <div className="flex flex-wrap items-center gap-2 text-xs text-white/45">
            <Brain className="size-4 text-amber-300/70" />
            {settings.memory.facts.length} fact(s) remembered
            <span className="font-mono text-[10px] text-white/25">
              {settings.memory.path}
            </span>
          </div>

          <textarea
            value={memoryDraft}
            onChange={(event) => setMemoryDraft(event.target.value)}
            rows={10}
            spellCheck={false}
            aria-label="Memory contents"
            className="w-full resize-y rounded-xl border border-white/10 bg-black/30 px-4 py-3 font-mono text-xs leading-relaxed text-white/75 outline-none placeholder:text-white/20 focus:border-white/20"
            placeholder="# LOMA memory&#10;&#10;- Name: Hamish&#10;- Lives in Mackay, Queensland"
          />

          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="max-w-sm text-[11px] leading-relaxed text-white/30">
              Edit it by hand, delete a line, or say{" "}
              <span className="text-white/50">“remember that …”</span> in chat.
              Auto-extraction is {settings.memory.auto_extract ? "on" : "off"}.
            </p>
            <div className="flex gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => void wipeMemory()}
                disabled={busy !== null}
                className="text-white/40 hover:text-red-400"
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
                className="bg-amber-300 text-black hover:bg-amber-200"
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
            <div className="flex items-center gap-2 text-xs text-white/45">
              <StatusDot on={settings.mcp.connected} />
              {settings.mcp.connected
                ? `${settings.mcp.tool_count} tools across ${settings.mcp.servers.length} server(s)`
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
              <p className="rounded-lg border border-dashed border-white/10 px-4 py-6 text-center text-xs text-white/30">
                No servers configured yet.
              </p>
            )}

            {settings.mcp.servers.map((server: McpServer) => (
              <div
                key={server.name}
                className="rounded-xl border border-white/10 bg-black/20 px-4 py-3"
              >
                <div className="flex items-start gap-3">
                  <Server className="mt-0.5 size-4 shrink-0 text-white/30" />
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-mono text-sm text-white/90">
                        {server.name}
                      </span>
                      <Badge
                        variant="secondary"
                        className="bg-white/5 text-[10px] font-normal text-white/40"
                      >
                        {server.transport}
                      </Badge>
                      <span className="flex items-center gap-1.5 font-mono text-[10px] text-white/35">
                        <StatusDot on={server.connected} />
                        {server.connected
                          ? `${server.tools.length} tool(s)`
                          : "offline"}
                      </span>
                    </div>
                    <p className="mt-1 truncate font-mono text-[11px] text-white/30">
                      {server.transport === "http"
                        ? server.url
                        : [server.command, ...server.args].join(" ")}
                    </p>
                    {server.tools.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-1.5">
                        {server.tools.map((tool) => (
                          <span
                            key={tool}
                            className="rounded bg-white/5 px-1.5 py-0.5 font-mono text-[10px] text-white/45"
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
                    className="text-white/30 hover:text-red-400"
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

          <Separator className="bg-white/5" />

          <div className="space-y-3">
            <p className="flex items-center gap-2 text-xs font-medium text-white/60">
              <Plus className="size-3.5" /> Add a server
            </p>

            <div className="grid gap-3 sm:grid-cols-2">
              <div className="grid gap-1.5">
                <Label htmlFor="server-name" className="text-[11px] text-white/45">
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
                <Label className="text-[11px] text-white/45">Transport</Label>
                <div className="flex gap-1 rounded-lg border border-white/10 p-1">
                  {(["stdio", "http"] as const).map((kind) => (
                    <button
                      key={kind}
                      onClick={() => setTransport(kind)}
                      className={cn(
                        "flex-1 rounded-md px-2 py-1 font-mono text-[11px] transition",
                        transport === kind
                          ? "bg-white/10 text-white"
                          : "text-white/40 hover:text-white/70",
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
                  <Label htmlFor="server-command" className="text-[11px] text-white/45">
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
                  <Label htmlFor="server-args" className="text-[11px] text-white/45">
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
                <Label htmlFor="server-url" className="text-[11px] text-white/45">
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
              <Label htmlFor="server-env" className="text-[11px] text-white/45">
                Environment (optional JSON)
              </Label>
              <textarea
                id="server-env"
                aria-label="Server environment variables as JSON"
                value={env}
                onChange={(event) => setEnv(event.target.value)}
                rows={2}
                placeholder={'{"API_KEY": "..."}'}
                className="w-full rounded-lg border border-white/10 bg-transparent px-3 py-2 font-mono text-xs text-white/80 outline-none placeholder:text-white/20 focus:border-white/20"
              />
            </div>

            <div className="flex items-center justify-between gap-3">
              <p className="font-mono text-[10px] text-white/25">
                saved to {settings.mcp.path}
              </p>
              <Button
                onClick={() => void addServer()}
                disabled={busy !== null || !name.trim() || (!command.trim() && !url.trim())}
                className="bg-amber-300 text-black hover:bg-amber-200"
              >
                {busy === "add-server" ? (
                  <Loader2 className="size-4 animate-spin" />
                ) : (
                  <Plus className="size-4" />
                )}
                Add server
              </Button>
            </div>
            <p className="text-[11px] leading-relaxed text-white/30">
              LOMA restarts the tool router after every change and rolls the file
              back if the server does not come up. <code>{"${LOMA_ROOT}"}</code> and{" "}
              <code>{"${LOMA_PYTHON}"}</code> are expanded when the server starts.
            </p>
          </div>
        </Section>

        <p className="pb-6 text-center font-mono text-[10px] text-white/25">
          {settings.context.policy}
        </p>
      </div>
    </div>
  )
}
