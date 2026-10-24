"use client"

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import { cn } from "@/lib/utils"
import { Check, ChevronDown, Loader2, Wrench, X } from "lucide-react"
import { useState } from "react"

export type ToolPart = {
  type: string
  state:
    | "input-streaming"
    | "input-available"
    | "output-available"
    | "output-error"
  input?: Record<string, unknown>
  output?: Record<string, unknown>
  toolCallId?: string
  errorText?: string
}

export type ToolProps = {
  toolPart: ToolPart
  defaultOpen?: boolean
  className?: string
}

/** One compact line per call. The interesting part is the result, not the chrome. */
const Tool = ({ toolPart, defaultOpen = false, className }: ToolProps) => {
  const [isOpen, setIsOpen] = useState(defaultOpen)
  const { state, input, output, toolCallId, errorText } = toolPart

  const running = state === "input-streaming" || state === "input-available"
  const failed = state === "output-error"

  const args = input && Object.keys(input).length > 0 ? input : null

  return (
    <div className={cn("overflow-hidden rounded-lg border border-loma-line bg-white/[0.02]", className)}>
      <Collapsible open={isOpen} onOpenChange={setIsOpen}>
        <CollapsibleTrigger
          className="flex w-full items-center gap-2.5 px-3 py-2 text-left transition hover:bg-white/[0.03]"
          aria-label={`${toolPart.type} — ${running ? "running" : failed ? "failed" : "completed"}`}
        >
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

          <span className="font-mono text-[12.5px] text-loma-ink">{toolPart.type}</span>

          {args && (
            <span className="truncate font-mono text-[11.5px] text-loma-faint">
              {Object.entries(args)
                .map(([, value]) =>
                  typeof value === "string" ? value : JSON.stringify(value),
                )
                .join(" · ")}
            </span>
          )}

          <span className="ml-auto flex shrink-0 items-center gap-2">
            <span
              className={cn(
                "font-mono text-[10px] uppercase tracking-[0.12em]",
                running && "text-loma-teal",
                !running && !failed && "text-loma-faint",
                failed && "text-loma-danger",
              )}
            >
              {running ? "running" : failed ? "failed" : "done"}
            </span>
            <ChevronDown
              className={cn(
                "size-3.5 text-loma-faint transition-transform duration-200",
                isOpen && "rotate-180",
              )}
            />
          </span>
        </CollapsibleTrigger>

        <CollapsibleContent className="overflow-hidden data-[state=closed]:animate-collapsible-up data-[state=open]:animate-collapsible-down">
          <div className="space-y-2.5 border-t border-loma-line-soft px-3 py-2.5">
            {args && (
              <div>
                <p className="mb-1 font-mono text-[10px] uppercase tracking-[0.14em] text-loma-faint">
                  arguments
                </p>
                <pre className="overflow-x-auto rounded-md bg-black/30 px-2.5 py-2 font-mono text-[11.5px] leading-relaxed text-loma-dim">
                  {JSON.stringify(args, null, 2)}
                </pre>
              </div>
            )}

            {output && (
              <div>
                <p className="mb-1 font-mono text-[10px] uppercase tracking-[0.14em] text-loma-faint">
                  result
                </p>
                <pre className="max-h-64 overflow-auto whitespace-pre-wrap rounded-md bg-black/30 px-2.5 py-2 font-mono text-[11.5px] leading-relaxed text-loma-ink/85">
                  {typeof output === "object"
                    ? JSON.stringify(output, null, 2)
                    : String(output)}
                </pre>
              </div>
            )}

            {failed && errorText && (
              <p className="rounded-md border border-loma-danger/25 bg-loma-danger/[0.06] px-2.5 py-2 font-mono text-[11.5px] text-loma-danger">
                {errorText}
              </p>
            )}

            {toolCallId && (
              <p className="flex items-center gap-1.5 font-mono text-[10px] text-loma-faint/70">
                <Wrench className="size-2.5" />
                {toolCallId}
              </p>
            )}
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  )
}

export { Tool }
