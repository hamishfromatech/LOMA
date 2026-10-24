import type { TurnMetrics as Metrics } from "@/lib/types"

type Props = {
  metrics: Metrics
  intent?: string
  matched?: string
}

function Item({ label, value }: { label: string; value: string }) {
  return (
    <span className="flex items-center gap-1">
      <span className="text-white/30">{label}</span>
      <span className="tabular-nums text-white/60">{value}</span>
    </span>
  )
}

/** The timing line under an answer — the part students should stare at. */
export function TurnMetrics({ metrics, intent, matched }: Props) {
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[10px]">
      <Item label="route" value={`${metrics.routeMs.toFixed(2)} ms`} />
      {metrics.toolMs > 0 && (
        <Item label="tool" value={`${Math.round(metrics.toolMs)} ms`} />
      )}
      <Item label="first token" value={`${metrics.firstTokenS.toFixed(2)} s`} />
      <Item label="total" value={`${metrics.totalS.toFixed(2)} s`} />
      <Item label="ctx" value={`${metrics.contextTokens} tok`} />
      {intent && intent !== "answer" && (
        <span className="rounded-full bg-amber-400/10 px-2 py-0.5 text-amber-300">
          {intent}
          {matched ? ` · "${matched}"` : ""}
        </span>
      )}
    </div>
  )
}
