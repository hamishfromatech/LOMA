import type { ToolUiItem, ToolUiSpec } from "./types"

/**
 * Tool results arrive as strings from MCP. A result may carry a presentation
 * hint under `ui` (or `_loma_ui`):
 *
 *   { "roll": 14, "sides": 20, "ui": { "type": "metric", ... } }
 *
 * The data stays data; the hint is validated here and rendered by a small
 * whitelist of primitives in `components/tool-cards.tsx`. Nothing in the spec
 * can become markup, a URL, a style or a script.
 */

const MAX_ITEMS = 12
const MAX_ROWS = 20
const MAX_COLUMNS = 8
const MAX_TEXT = 240
const MAX_CODE = 4000

function text(value: unknown, limit = MAX_TEXT): string {
  if (typeof value === "string") return value.slice(0, limit)
  if (typeof value === "number" || typeof value === "boolean") return String(value)
  return ""
}

function finite(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined
}

function item(raw: unknown): ToolUiItem | null {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null
  const record = raw as Record<string, unknown>
  const label = text(record.label)
  const value = text(record.value)
  if (!label || !value) return null
  const hint = text(record.hint)
  const percent = finite(record.percent)
  return {
    label,
    value,
    ...(hint ? { hint } : {}),
    ...(percent !== undefined ? { percent: Math.min(100, Math.max(0, percent)) } : {}),
  }
}

export function sanitizeUiSpec(raw: unknown): ToolUiSpec | null {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null
  const spec = raw as Record<string, unknown>
  const type = text(spec.type)
  const title = text(spec.title) || undefined
  const subtitle = text(spec.subtitle) || undefined

  switch (type) {
    case "metric":
    case "keyvalue": {
      const items = Array.isArray(spec.items)
        ? spec.items.slice(0, MAX_ITEMS).map(item).filter((entry): entry is ToolUiItem => entry !== null)
        : []
      if (items.length === 0) return null
      return type === "metric"
        ? { type, title, subtitle, items }
        : { type, title, items }
    }
    case "table": {
      const columns = Array.isArray(spec.columns)
        ? spec.columns.slice(0, MAX_COLUMNS).map((column) => text(column)).filter(Boolean)
        : []
      const rows = Array.isArray(spec.rows)
        ? spec.rows
            .slice(0, MAX_ROWS)
            .map((row) =>
              Array.isArray(row)
                ? row.slice(0, columns.length || MAX_COLUMNS).map((cell) => text(cell))
                : [],
            )
            .filter((row) => row.some(Boolean))
        : []
      if (columns.length === 0 || rows.length === 0) return null
      return { type, title, columns, rows }
    }
    case "list": {
      const items = Array.isArray(spec.items)
        ? spec.items.slice(0, MAX_ITEMS).map((entry) => text(entry)).filter(Boolean)
        : []
      if (items.length === 0) return null
      return { type, title, items }
    }
    case "code": {
      const code = text(spec.code, MAX_CODE)
      if (!code) return null
      return {
        type,
        title,
        language: text(spec.language) || undefined,
        code,
      }
    }
    case "progress": {
      const label = text(spec.label)
      const value = finite(spec.value)
      if (!label || value === undefined) return null
      const max = finite(spec.max)
      return {
        type,
        title,
        label,
        value,
        ...(max !== undefined && max > 0 ? { max } : {}),
        unit: text(spec.unit) || undefined,
      }
    }
    default:
      return null
  }
}

export type ParsedToolResult = {
  /** The tool's data, with any UI hint removed. */
  data: unknown
  ui?: ToolUiSpec
}

export function parseToolResult(raw?: string): ParsedToolResult {
  if (!raw) return { data: undefined }
  const trimmed = raw.trim()
  try {
    const parsed: unknown = JSON.parse(trimmed)
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      const record = parsed as Record<string, unknown>
      const ui = sanitizeUiSpec(record.ui ?? record._loma_ui)
      if (ui) {
        const data = { ...record }
        delete data.ui
        delete data._loma_ui
        return { data: Object.keys(data).length > 0 ? data : undefined, ui }
      }
    }
    return { data: parsed }
  } catch {
    return { data: trimmed }
  }
}
