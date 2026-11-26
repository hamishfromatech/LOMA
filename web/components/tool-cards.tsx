"use client"

import type { ToolCallView, ToolUiItem, ToolUiSpec } from "@/lib/types"
import { parseToolResult } from "@/lib/tool-ui"
import { cn } from "@/lib/utils"
import {
  Calculator,
  Check,
  ChevronDown,
  Clock3,
  Cpu,
  Dices,
  Loader2,
  Wrench,
  X,
} from "lucide-react"
import type { ReactNode } from "react"

/**
 * Generative tool UI.
 *
 * A tool result can arrive in three forms, in this order of preference:
 *
 * 1. a declarative `ui` spec from the tool itself (validated in lib/tool-ui.ts);
 * 2. a bespoke component registered by tool name (the demo tools below);
 * 3. a generic key/value card.
 *
 * The card never renders markup from the tool. The spec is data, the renderers
 * are ours.
 */

const ICONS: Record<string, typeof Wrench> = {
  get_current_time: Clock3,
  roll_die: Dices,
  calculate: Calculator,
  system_stats: Cpu,
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null
}

function scalar(value: unknown): string {
  if (typeof value === "string") return value
  if (typeof value === "number" || typeof value === "boolean") return String(value)
  if (Array.isArray(value)) return value.map((entry) => scalar(entry)).join(", ")
  if (value && typeof value === "object") return JSON.stringify(value)
  return ""
}

function pretty(raw?: string): string {
  if (!raw) return ""
  try {
    return JSON.stringify(JSON.parse(raw), null, 2)
  } catch {
    return raw
  }
}

function valueOf(data: unknown): string {
  return typeof data === "string" ? data : (asRecord(data)?.time as string) ?? ""
}

// ---------------------------------------------------------------------------
// The frame: one header, then whatever the result deserves
// ---------------------------------------------------------------------------

function ToolFrame({
  tool,
  icon,
  children,
}: {
  tool: ToolCallView
  icon: ReactNode
  children: ReactNode
}) {
  const running = tool.status === "running"
  const failed = tool.status === "error"
  const hasDetails = Boolean(tool.result) || Object.keys(tool.arguments).length > 0

  return (
    <div className="loma-rise overflow-hidden rounded-xl border border-loma-line bg-white/[0.02]">
      <div className="flex items-center gap-2.5 px-3 py-2">
        <span
          className={cn(
            "grid size-4 shrink-0 place-items-center rounded-full",
            running && "text-loma-teal",
            !running && !failed && "text-loma-ok",
            failed && "text-loma-danger",
          )}
        >
          {running ? (
            <Loader2 className="size-3.5 animate-spin" />
          ) : failed ? (
            <X className="size-3.5" />
          ) : (
            <Check className="size-3.5" />
          )}
        </span>

        <span className="flex items-center gap-1.5 font-mono text-[12.5px] text-loma-ink">
          <span className="text-loma-faint">{icon}</span>
          {tool.name}
        </span>

        {!running && tool.latencyMs !== undefined && (
          <span className="font-mono text-[10px] tabular-nums text-loma-faint">
            {Math.round(tool.latencyMs)} ms
          </span>
        )}

        <span
          className={cn(
            "ml-auto font-mono text-[10px] uppercase tracking-[0.12em]",
            running && "text-loma-teal",
            !running && !failed && "text-loma-faint",
            failed && "text-loma-danger",
          )}
        >
          {running ? "running" : failed ? "failed" : "done"}
        </span>
      </div>

      {children}

      {hasDetails && (
        <details className="group border-t border-loma-line-soft">
          <summary className="flex cursor-pointer list-none items-center gap-1 px-3 py-1.5 font-mono text-[10px] uppercase tracking-[0.14em] text-loma-faint transition hover:text-loma-dim">
            <ChevronDown className="size-3 transition-transform group-open:rotate-180" />
            raw
          </summary>
          <div className="space-y-2 px-3 pb-2.5">
            {Object.keys(tool.arguments).length > 0 && (
              <div>
                <p className="mb-1 font-mono text-[10px] uppercase tracking-[0.14em] text-loma-faint">
                  arguments
                </p>
                <pre className="overflow-x-auto rounded-md bg-black/30 px-2.5 py-2 font-mono text-[11.5px] leading-relaxed text-loma-dim">
                  {JSON.stringify(tool.arguments, null, 2)}
                </pre>
              </div>
            )}
            {tool.result && (
              <div>
                <p className="mb-1 font-mono text-[10px] uppercase tracking-[0.14em] text-loma-faint">
                  result
                </p>
                <pre className="max-h-64 overflow-auto whitespace-pre-wrap rounded-md bg-black/30 px-2.5 py-2 font-mono text-[11.5px] leading-relaxed text-loma-ink/85">
                  {pretty(tool.result)}
                </pre>
              </div>
            )}
          </div>
        </details>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Bespoke cards for the demo tools
// ---------------------------------------------------------------------------

function TimeCard({ data }: { data: unknown }) {
  const text = valueOf(data)
  if (!text) return <GenericCard data={data} />
  const [time, ...rest] = text.split(" on ")

  return (
    <div className="flex items-center gap-4 px-4 pb-4 pt-1">
      <div className="grid size-14 shrink-0 place-items-center rounded-full border border-loma-teal/25 bg-loma-teal/10 text-loma-teal">
        <Clock3 className="size-6" />
      </div>
      <div className="min-w-0">
        <p className="text-[26px] font-medium leading-tight tracking-[-0.02em] text-loma-ink">
          {time}
        </p>
        {rest.length > 0 && (
          <p className="mt-0.5 truncate text-[12.5px] text-loma-dim">
            on {rest.join(" on ")}
          </p>
        )}
      </div>
    </div>
  )
}

function DieCard({ tool, data }: { tool: ToolCallView; data: unknown }) {
  const record = asRecord(data)
  const roll = Number(record?.roll)
  const sides = Number(record?.sides ?? tool.arguments.sides) || 6
  if (!Number.isFinite(roll)) return <GenericCard data={data} />

  return (
    <div className="flex items-center gap-4 px-4 pb-4 pt-1">
      <div className="loma-pop grid size-[68px] shrink-0 place-items-center rounded-2xl border border-loma-accent/30 bg-gradient-to-br from-loma-accent/[0.18] to-transparent text-[34px] font-semibold tabular-nums leading-none text-loma-ink shadow-[0_0_0_1px_rgba(255,255,255,0.02)_inset,0_18px_40px_-24px_rgba(0,0,0,0.9)]">
        {roll}
      </div>
      <div>
        <p className="text-[13px] text-loma-dim">
          rolled a <span className="font-medium text-loma-ink">d{sides}</span>
        </p>
        <p className="mt-0.5 font-mono text-[10.5px] text-loma-faint">
          1–{sides}
        </p>
      </div>
    </div>
  )
}

function CalcCard({ tool, data }: { tool: ToolCallView; data: unknown }) {
  const record = asRecord(data)
  const expression =
    (record?.expression as string) ||
    (tool.arguments.expression as string) ||
    "expression"
  const result = record?.result ?? record?.value ?? data

  return (
    <div className="px-4 pb-4 pt-1">
      <p className="truncate font-mono text-[12px] text-loma-faint">{expression}</p>
      <div className="mt-1 flex items-baseline gap-2.5">
        <span className="text-[18px] text-loma-faint">=</span>
        <span className="text-[30px] font-medium leading-none tabular-nums tracking-[-0.02em] text-loma-ink">
          {scalar(result)}
        </span>
      </div>
    </div>
  )
}

function Stat({
  label,
  value,
  className,
}: {
  label: string
  value: string
  className?: string
}) {
  return (
    <div className={cn("rounded-lg bg-black/20 px-3 py-2", className)}>
      <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-loma-faint">
        {label}
      </p>
      <p className="mt-0.5 break-words text-[13px] text-loma-ink">{value || "—"}</p>
    </div>
  )
}

function StatsCard({ data }: { data: unknown }) {
  const record = asRecord(data)
  if (!record) return <GenericCard data={data} />
  const used = Math.min(100, Math.max(0, Number(record.memory_used_percent) || 0))
  const available = Number(record.memory_available_mb)
  const cores = Number(record.cpu_cores)

  return (
    <div className="grid gap-2 px-4 pb-4 pt-1 sm:grid-cols-2">
      <Stat label="system" value={scalar(record.os)} />
      <Stat label="cpu cores" value={Number.isFinite(cores) ? String(cores) : ""} />
      <div className="rounded-lg bg-black/20 px-3 py-2 sm:col-span-2">
        <div className="flex items-center justify-between gap-3">
          <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-loma-faint">
            memory used
          </p>
          <p className="font-mono text-[11px] tabular-nums text-loma-dim">
            {used.toFixed(1)}%
          </p>
        </div>
        <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-white/[0.06]">
          <div
            className="h-full rounded-full bg-loma-teal transition-[width] duration-500"
            style={{ width: `${used}%` }}
          />
        </div>
        {Number.isFinite(available) && (
          <p className="mt-1 font-mono text-[10.5px] text-loma-faint">
            {Math.round(available).toLocaleString()} MB free
          </p>
        )}
      </div>
    </div>
  )
}

function GenericCard({ data }: { data: unknown }) {
  const record = asRecord(data)
  if (!record) {
    const text = scalar(data)
    return (
      <p className="px-4 pb-4 pt-1 text-[13px] leading-relaxed text-loma-ink">
        {text || "done"}
      </p>
    )
  }

  return (
    <dl className="grid gap-2 px-4 pb-4 pt-1 sm:grid-cols-2">
      {Object.entries(record)
        .slice(0, 12)
        .map(([key, value]) => (
          <div key={key} className="rounded-lg bg-black/20 px-3 py-2">
            <dt className="font-mono text-[10px] uppercase tracking-[0.12em] text-loma-faint">
              {key.replace(/_/g, " ")}
            </dt>
            <dd className="mt-0.5 break-words text-[13px] text-loma-ink">
              {scalar(value) || "—"}
            </dd>
          </div>
        ))}
    </dl>
  )
}

// ---------------------------------------------------------------------------
// Declarative cards: a tool's own ui spec
// ---------------------------------------------------------------------------

function CardTitle({ spec }: { spec: ToolUiSpec }) {
  if (!("title" in spec) || (!spec.title && !("subtitle" in spec && spec.subtitle))) {
    return null
  }
  return (
    <div className="px-4 pb-1 pt-1">
      {spec.title && <p className="text-[13px] font-medium text-loma-ink">{spec.title}</p>}
      {"subtitle" in spec && spec.subtitle && (
        <p className="text-[11.5px] text-loma-faint">{spec.subtitle}</p>
      )}
    </div>
  )
}

function MetricItems({ items }: { items: ToolUiItem[] }) {
  return (
    <div className="grid gap-2 sm:grid-cols-2">
      {items.map((item) => (
        <div key={`${item.label}-${item.value}`} className="rounded-lg bg-black/20 px-3 py-2">
          <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-loma-faint">
            {item.label}
          </p>
          <p className="mt-0.5 break-words text-[15px] text-loma-ink">{item.value}</p>
          {item.hint && <p className="text-[10.5px] text-loma-faint">{item.hint}</p>}
          {item.percent !== undefined && (
            <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-white/[0.06]">
              <div
                className="h-full rounded-full bg-loma-teal"
                style={{ width: `${item.percent}%` }}
              />
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

function DeclarativeCard({ spec }: { spec: ToolUiSpec }) {
  return (
    <div className="space-y-2 px-4 pb-4 pt-1">
      <CardTitle spec={spec} />
      {spec.type === "metric" && <MetricItems items={spec.items} />}
      {spec.type === "keyvalue" && (
        <dl className="grid gap-2 sm:grid-cols-2">
          {spec.items.map((item) => (
            <div key={`${item.label}-${item.value}`} className="rounded-lg bg-black/20 px-3 py-2">
              <dt className="font-mono text-[10px] uppercase tracking-[0.12em] text-loma-faint">
                {item.label}
              </dt>
              <dd className="mt-0.5 break-words text-[13px] text-loma-ink">{item.value}</dd>
            </div>
          ))}
        </dl>
      )}
      {spec.type === "table" && (
        <div className="overflow-x-auto rounded-lg bg-black/20">
          <table className="w-full border-collapse text-left text-[12px]">
            <thead>
              <tr className="border-b border-loma-line-soft">
                {spec.columns.map((column) => (
                  <th
                    key={column}
                    className="px-3 py-1.5 font-mono text-[10px] uppercase tracking-[0.1em] text-loma-faint"
                  >
                    {column}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {spec.rows.map((row, index) => (
                <tr key={index} className="border-b border-loma-line-soft/60 last:border-0">
                  {row.map((cell, cellIndex) => (
                    <td key={cellIndex} className="px-3 py-1.5 text-loma-ink">
                      {cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {spec.type === "list" && (
        <ul className="flex flex-wrap gap-1.5">
          {spec.items.map((item, index) => (
            <li
              key={`${item}-${index}`}
              className="rounded-full border border-loma-line bg-black/20 px-2.5 py-1 text-[12px] text-loma-dim"
            >
              {item}
            </li>
          ))}
        </ul>
      )}
      {spec.type === "code" && (
        <pre className="max-h-72 overflow-auto rounded-lg bg-black/30 px-3 py-2 font-mono text-[11.5px] leading-relaxed text-loma-ink/90">
          {spec.code}
        </pre>
      )}
      {spec.type === "progress" && (
        <div className="rounded-lg bg-black/20 px-3 py-2">
          <div className="flex items-center justify-between gap-3">
            <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-loma-faint">
              {spec.label}
            </p>
            <p className="font-mono text-[11px] tabular-nums text-loma-dim">
              {spec.value}
              {spec.unit ?? ""}
              {spec.max ? ` / ${spec.max}${spec.unit ?? ""}` : ""}
            </p>
          </div>
          <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-white/[0.06]">
            <div
              className="h-full rounded-full bg-loma-teal"
              style={{
                width: `${Math.min(100, Math.max(0, (spec.value / (spec.max ?? 100)) * 100))}%`,
              }}
            />
          </div>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Running / failed
// ---------------------------------------------------------------------------

function RunningCard({ tool }: { tool: ToolCallView }) {
  const args = Object.entries(tool.arguments)
  return (
    <div className="space-y-1.5 px-4 pb-4 pt-1">
      {args.length > 0 ? (
        <p className="truncate font-mono text-[11.5px] text-loma-faint">
          {args
            .map(([, value]) => (typeof value === "string" ? value : JSON.stringify(value)))
            .join(" · ")}
        </p>
      ) : (
        <p className="font-mono text-[11.5px] text-loma-faint">working…</p>
      )}
    </div>
  )
}

function FailedCard({ tool }: { tool: ToolCallView }) {
  return (
    <p className="mx-4 mb-4 mt-1 rounded-md border border-loma-danger/25 bg-loma-danger/[0.06] px-2.5 py-2 font-mono text-[11.5px] text-loma-danger">
      {tool.result || "the tool failed"}
    </p>
  )
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

type RendererProps = { tool: ToolCallView; data: unknown }

const TOOL_RENDERERS: Record<string, (props: RendererProps) => ReactNode> = {
  get_current_time: ({ data }) => <TimeCard data={data} />,
  roll_die: ({ tool, data }) => <DieCard tool={tool} data={data} />,
  calculate: ({ tool, data }) => <CalcCard tool={tool} data={data} />,
  system_stats: ({ data }) => <StatsCard data={data} />,
}

export function ToolResultCard({ tool }: { tool: ToolCallView }) {
  const { data, ui } = parseToolResult(tool.result)
  const Icon = ICONS[tool.name] ?? Wrench

  let body: ReactNode
  if (tool.status === "running") {
    body = <RunningCard tool={tool} />
  } else if (tool.status === "error") {
    body = <FailedCard tool={tool} />
  } else if (ui) {
    body = <DeclarativeCard spec={ui} />
  } else {
    const Renderer = TOOL_RENDERERS[tool.name]
    body = Renderer ? <Renderer tool={tool} data={data} /> : <GenericCard data={data} />
  }

  return (
    <ToolFrame tool={tool} icon={<Icon className="size-3.5" />}>
      {body}
    </ToolFrame>
  )
}
