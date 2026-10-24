"use client"

import { cn } from "@/lib/utils"
import { Check, Copy } from "lucide-react"
import React, { useEffect, useState } from "react"
import { codeToHtml } from "shiki"

export type CodeBlockProps = {
  children?: React.ReactNode
  className?: string
} & React.HTMLProps<HTMLDivElement>

function CodeBlock({ children, className, ...props }: CodeBlockProps) {
  return (
    <div
      className={cn(
        "loma-code flex w-full flex-col overflow-hidden",
        "rounded-xl border border-loma-line bg-[#121110]",
        className,
      )}
      {...props}
    >
      {children}
    </div>
  )
}

export type CodeBlockCodeProps = {
  code: string
  language?: string
  theme?: string
  className?: string
} & React.HTMLProps<HTMLDivElement>

function CodeBlockCode({
  code,
  language = "plaintext",
  theme = "github-dark-dimmed",
  className,
  ...props
}: CodeBlockCodeProps) {
  const [highlightedHtml, setHighlightedHtml] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)
  const source = String(code ?? "").replace(/\n$/, "")

  useEffect(() => {
    let cancelled = false
    async function highlight() {
      if (!source.trim()) {
        setHighlightedHtml("<pre><code></code></pre>")
        return
      }
      try {
        const html = await codeToHtml(source, { lang: language, theme })
        if (!cancelled) setHighlightedHtml(html)
      } catch {
        // An unknown language should never blank the answer.
        if (!cancelled) setHighlightedHtml(null)
      }
    }
    highlight()
    return () => {
      cancelled = true
    }
  }, [source, language, theme])

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(source)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1400)
    } catch {
      /* clipboard blocked — the code is still selectable */
    }
  }

  return (
    <div className={cn("w-full", className)} {...props}>
      <div className="flex items-center justify-between border-b border-loma-line-soft px-3 py-1.5">
        <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-loma-faint">
          {language}
        </span>
        <button
          type="button"
          onClick={() => void copy()}
          aria-label={copied ? "Copied" : "Copy code"}
          className="flex items-center gap-1.5 rounded-md px-1.5 py-1 font-mono text-[10px] uppercase tracking-[0.1em] text-loma-faint transition hover:bg-white/[0.06] hover:text-loma-ink"
        >
          {copied ? <Check className="size-3 text-loma-ok" /> : <Copy className="size-3" />}
          {copied ? "copied" : "copy"}
        </button>
      </div>

      {highlightedHtml ? (
        <div
          className="w-full overflow-x-auto"
          dangerouslySetInnerHTML={{ __html: highlightedHtml }}
        />
      ) : (
        <div className="w-full overflow-x-auto">
          <pre className="p-4 font-mono text-[13px] leading-relaxed text-loma-ink">
            <code>{source}</code>
          </pre>
        </div>
      )}
    </div>
  )
}

export type CodeBlockGroupProps = React.HTMLAttributes<HTMLDivElement>

function CodeBlockGroup({ children, className, ...props }: CodeBlockGroupProps) {
  return (
    <div className={cn("flex items-center justify-between", className)} {...props}>
      {children}
    </div>
  )
}

export { CodeBlockGroup, CodeBlockCode, CodeBlock }
