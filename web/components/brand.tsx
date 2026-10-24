import { cn } from "@/lib/utils"
import { Waves } from "lucide-react"

/**
 * The mark: a cream tile with a warm ink wave, lifted from the cover art.
 * Deliberately the only high-contrast element in the chrome.
 */
export function LomaMark({
  size = 28,
  className,
  tone = "solid",
}: {
  size?: number
  className?: string
  tone?: "solid" | "quiet"
}) {
  return (
    <span
      style={{ width: size, height: size, borderRadius: Math.round(size * 0.32) }}
      className={cn(
        "grid shrink-0 place-items-center",
        tone === "solid"
          ? "bg-[#f4efe7] text-[#1b1713] shadow-[0_1px_0_rgba(255,255,255,0.35)_inset,0_4px_12px_-6px_rgba(0,0,0,0.9)]"
          : "border border-loma-line bg-white/[0.04] text-loma-accent",
        className,
      )}
    >
      <Waves style={{ width: size * 0.56, height: size * 0.56 }} strokeWidth={2.1} />
    </span>
  )
}

export function Wordmark({ className }: { className?: string }) {
  return (
    <span
      className={cn(
        "font-mono text-[11px] uppercase tracking-[0.22em] text-loma-ink",
        className,
      )}
    >
      Loma
    </span>
  )
}
