"use client"

import { cn } from "@/lib/utils"
import type { Status } from "@/lib/types"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { Cpu, HardDrive, Layers, Wrench } from "lucide-react"

type Props = {
  status: Status | null
  className?: string
}

function Chip({
  icon,
  label,
  value,
  tone = "default",
  children,
}: {
  icon: React.ReactNode
  label: string
  value: string
  tone?: "default" | "warn" | "ok"
  children?: React.ReactNode
}) {
  return (
    <div className="flex items-center gap-2 rounded-full border border-loma-line bg-white/[0.03] px-2.5 py-1">
      <span className="text-loma-faint">{icon}</span>
      <span className="font-mono text-[10.5px] tracking-tight text-loma-faint">
        {label}
      </span>
      <span
        className={cn(
          "font-mono text-[10.5px] tabular-nums",
          tone === "warn" && "text-loma-accent",
          tone === "ok" && "text-loma-ok",
          tone === "default" && "text-loma-ink",
        )}
      >
        {value}
      </span>
      {children}
    </div>
  )
}

export function StatusBar({ status, className }: Props) {
  if (!status) {
    return (
      <div className={cn("flex items-center gap-2", className)}>
        <Chip icon={<Cpu className="size-3" />} label="status" value="connecting" />
      </div>
    )
  }

  const used = Math.round(status.models.total_mb)
  const budget = status.memory_budget_mb
  const over = used > budget
  const tools = status.tools.length

  return (
    <div className={cn("flex flex-wrap items-center gap-2", className)}>
      <Chip
        icon={<Cpu className="size-3" />}
        label="llm"
        value={status.config.llm_model}
      />
      <Chip
        icon={<HardDrive className="size-3" />}
        label="ram"
        value={`${used} / ${budget} MB`}
        tone={over ? "warn" : "ok"}
      />
      <Chip
        icon={<Layers className="size-3" />}
        label="ctx"
        value={`${status.context.live_tokens} / ${Math.round(status.context.window / 1024)}k`}
      />
      <Tooltip>
        <TooltipTrigger
          render={
            <div className="flex cursor-default items-center gap-2 rounded-full border border-loma-line bg-white/[0.03] px-2.5 py-1">
              <Wrench className="size-3 text-loma-faint" />
              <span className="font-mono text-[10.5px] text-loma-faint">tools</span>
              <span className="font-mono text-[10.5px] tabular-nums text-loma-ink">
                {tools}
              </span>
            </div>
          }
        />
        <TooltipContent side="bottom" className="max-w-xs">
          <div className="space-y-1">
            {status.tools.length === 0 && <p>no MCP tools configured</p>}
            {status.tools.map((tool) => (
              <p key={tool.name} className="font-mono text-xs">
                {tool.name}
              </p>
            ))}
          </div>
        </TooltipContent>
      </Tooltip>
    </div>
  )
}
