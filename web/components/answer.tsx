"use client"

import { LomaMark } from "@/components/brand"
import { DotsLoader } from "@/components/ui/loader"
import { Markdown } from "@/components/ui/markdown"
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import { Tool, type ToolPart } from "@/components/ui/tool"
import { cn } from "@/lib/utils"
import type { ChatMessage, ToolCallView, TurnMetrics as Metrics } from "@/lib/types"
import { Check, ChevronRight, Copy, Brain, TriangleAlert } from "lucide-react"
import { useCallback, useState } from "react"

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

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <span className="flex items-baseline gap-1.5">
      <span className="text-loma-faint">{label}</span>
      <span className="tabular-nums text-loma-dim">{value}</span>
    </span>
  )
}

function Trace({ metrics, intent, matched }: { metrics: Metrics; intent?: string; matched?: string }) {
  const [open, setOpen] = useState(false)
  const hasIntent = intent && intent !== "answer"

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 font-mono text-[10.5px]">
        <CollapsibleTrigger
          className="flex items-center gap-1 rounded text-loma-faint transition hover:text-loma-dim"
          aria-label="Show the trace for this answer"
        >
          <ChevronRight
            className={cn("size-3 transition-transform duration-150", open && "rotate-90")}
          />
          <span className="tabular-nums">{metrics.totalS.toFixed(2)} s</span>
          <span className="text-loma-faint/50">·</span>
          <span className="tabular-nums">
            {metrics.contextTokens.toLocaleString()} tok
          </span>
        </CollapsibleTrigger>

        {hasIntent && (
          <span className="rounded-full bg-loma-accent/10 px-2 py-0.5 text-loma-accent/90">
            {intent}
            {matched ? ` · “${matched}”` : ""}
          </span>
        )}
      </div>

      <CollapsibleContent className="overflow-hidden data-[state=closed]:animate-collapsible-up data-[state=open]:animate-collapsible-down">
        <div className="mt-2 space-y-1 rounded-lg border border-loma-line-soft bg-black/20 px-3 py-2 font-mono text-[10.5px]">
          <div className="flex flex-wrap gap-x-4 gap-y-1">
            <Stat label="route" value={`${metrics.routeMs.toFixed(2)} ms`} />
            {metrics.toolMs > 0 && <Stat label="tool" value={`${Math.round(metrics.toolMs)} ms`} />}
            <Stat label="first token" value={`${metrics.firstTokenS.toFixed(2)} s`} />
            <Stat label="first audio" value={`${metrics.firstAudioS.toFixed(2)} s`} />
            <Stat label="total" value={`${metrics.totalS.toFixed(2)} s`} />
            <Stat label="context" value={`${metrics.contextTokens.toLocaleString()} tok`} />
          </div>
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}

export function AnswerMessage({ message }: { message: ChatMessage }) {
  const [copied, setCopied] = useState(false)

  const copy = useCallback(async () => {
    if (!message.content) return
    try {
      await navigator.clipboard.writeText(message.content)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1400)
    } catch {
      /* clipboard blocked */
    }
  }, [message.content])

  const thinking = message.streaming && !message.content
  const stage = message.intent && message.intent !== "answer" ? message.intent : "thinking"

  return (
    <article className="group/answer loma-rise flex gap-3">
      <LomaMark size={26} tone="quiet" className="mt-[3px]" />

      <div className="min-w-0 flex-1 space-y-2.5">
        {message.tools.map((tool) => (
          <Tool key={tool.id} toolPart={toToolPart(tool)} />
        ))}

        {thinking && (
          <div className="flex items-center gap-2.5 py-1.5" aria-live="polite">
            <DotsLoader size="sm" className="[&>div]:bg-loma-accent/80" />
            <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-loma-faint">
              {stage}
            </span>
          </div>
        )}

        {message.content && (
          <div className={cn("loma-prose", message.streaming && "loma-caret")} aria-live="polite">
            <Markdown>{message.content}</Markdown>
          </div>
        )}

        {message.memory && (
          <div className="flex items-start gap-2 rounded-lg border border-loma-accent/20 bg-loma-accent/[0.06] px-3 py-2 text-[12.5px] text-loma-accent/90">
            <Brain className="mt-0.5 size-3.5 shrink-0" />
            <span>
              Remembered <span className="text-loma-accent">{message.memory}</span>
            </span>
          </div>
        )}

        {message.error && (
          <div className="flex items-start gap-2 rounded-lg border border-loma-danger/25 bg-loma-danger/[0.06] px-3 py-2 text-[12.5px] text-loma-danger">
            <TriangleAlert className="mt-0.5 size-3.5 shrink-0" />
            <span>{message.error}</span>
          </div>
        )}

        {message.metrics && !message.streaming && (
          <div className="flex items-center justify-between gap-3 pt-0.5">
            <Trace
              metrics={message.metrics}
              intent={message.intent}
              matched={message.matched}
            />

            <button
              type="button"
              onClick={() => void copy()}
              aria-label={copied ? "Copied" : "Copy answer"}
              className={cn(
                "flex items-center gap-1.5 rounded-md px-1.5 py-1 font-mono text-[10px] uppercase tracking-[0.1em] transition",
                "text-loma-faint hover:bg-white/[0.05] hover:text-loma-ink",
                "opacity-100 sm:opacity-0 sm:group-hover/answer:opacity-100 sm:focus-visible:opacity-100",
              )}
            >
              {copied ? <Check className="size-3 text-loma-ok" /> : <Copy className="size-3" />}
              {copied ? "copied" : "copy"}
            </button>
          </div>
        )}
      </div>
    </article>
  )
}

export function UserMessage({ message }: { message: ChatMessage }) {
  return (
    <div className="loma-rise flex justify-end">
      <div className="max-w-[82%] whitespace-pre-wrap break-words rounded-[16px] rounded-br-[6px] border border-loma-line bg-loma-raise px-3.5 py-2.5 text-[14.5px] leading-relaxed text-loma-ink shadow-[0_1px_0_rgba(255,250,242,0.04)_inset]">
        {message.content}
      </div>
    </div>
  )
}
