"use client"

import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import type { ConversationSummary } from "@/lib/types"
import {
  Check,
  MessageSquare,
  Pencil,
  Plus,
  Search,
  Trash2,
  X,
} from "lucide-react"
import { useEffect, useMemo, useRef, useState } from "react"

type Props = {
  conversations: ConversationSummary[]
  activeId: string | null
  open: boolean
  onClose: () => void
  onNew: () => void
  onSelect: (id: string) => void
  onDelete: (id: string) => void
  onRename: (id: string, title: string) => void
  loading?: boolean
  error?: string | null
  onRetry?: () => void
  /** The button that would open the drawer, to return focus to on close. */
  triggerRef?: React.RefObject<HTMLElement | null>
}

const ORDER = [
  "Today",
  "Yesterday",
  "Previous 7 days",
  "Previous 30 days",
  "Older",
] as const

function dayKey(iso: string): (typeof ORDER)[number] {
  const date = new Date(iso)
  const now = new Date()
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate())
  const diff = (startOfToday.getTime() - date.getTime()) / 86_400_000
  if (diff < 0) return "Today"
  if (diff < 1) return "Yesterday"
  if (diff < 7) return "Previous 7 days"
  if (diff < 30) return "Previous 30 days"
  return "Older"
}

export function ChatsDrawer({
  conversations,
  activeId,
  open,
  onClose,
  onNew,
  onSelect,
  onDelete,
  onRename,
  loading = false,
  error = null,
  onRetry,
  triggerRef,
}: Props) {
  const [query, setQuery] = useState("")
  const [renaming, setRenaming] = useState<string | null>(null)
  const [draft, setDraft] = useState("")
  const [confirming, setConfirming] = useState<string | null>(null)
  const searchRef = useRef<HTMLInputElement>(null)
  const asideRef = useRef<HTMLElement>(null)
  const renameRef = useRef<HTMLInputElement>(null)
  const cancelledRef = useRef(false)
  const wasOpen = useRef(false)

  // Return focus to the rail control that opened the drawer when it closes.
  useEffect(() => {
    if (wasOpen.current && !open) triggerRef?.current?.focus()
    wasOpen.current = open
  }, [open, triggerRef])

  useEffect(() => {
    if (!open) {
      setRenaming(null)
      setConfirming(null)
      setQuery("")
      cancelledRef.current = false
      return
    }
    const timer = window.setTimeout(() => searchRef.current?.focus(), 180)
    return () => window.clearTimeout(timer)
  }, [open])

  // Trap Tab inside the drawer so focus can't escape behind the backdrop.
  useEffect(() => {
    if (!open) return
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Tab") return
      const aside = asideRef.current
      if (!aside) return
      const focusable = Array.from(
        aside.querySelectorAll<HTMLElement>(
          "button, input, textarea, select, a[href], [tabindex]:not([tabindex='-1'])",
        ),
      ).filter((element) => !element.hasAttribute("disabled"))
      if (focusable.length === 0) return
      const first = focusable[0]
      const last = focusable[focusable.length - 1]
      if (event.shiftKey && document.activeElement === first) {
        last.focus()
        event.preventDefault()
      } else if (!event.shiftKey && document.activeElement === last) {
        first.focus()
        event.preventDefault()
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [open])

  useEffect(() => {
    if (renaming) renameRef.current?.select()
  }, [renaming])

  const groups = useMemo(() => {
    const needle = query.trim().toLowerCase()
    const filtered = needle
      ? conversations.filter((item) => item.title.toLowerCase().includes(needle))
      : conversations

    const buckets = new Map<string, ConversationSummary[]>()
    for (const conversation of filtered) {
      const key = dayKey(conversation.updated_at)
      const list = buckets.get(key) ?? []
      list.push(conversation)
      buckets.set(key, list)
    }

    return ORDER.filter((key) => buckets.has(key)).map((key) => ({
      label: key,
      items: buckets.get(key)!,
    }))
  }, [conversations, query])

  const commitRename = (id: string) => {
    if (cancelledRef.current) {
      setRenaming(null)
      cancelledRef.current = false
      return
    }
    const next = draft.trim()
    if (next) onRename(id, next)
    setRenaming(null)
  }

  return (
    <>
      {open && (
        <div
          className="fixed inset-0 z-30 bg-black/55 backdrop-blur-[2px]"
          onClick={onClose}
          aria-hidden
        />
      )}

      <aside
        ref={asideRef}
        inert={!open}
        aria-hidden={!open}
        aria-label="Chats"
        role="dialog"
        aria-modal={open || undefined}
        className={cn(
          "fixed inset-y-3 left-12 z-40 flex w-[min(300px,calc(100vw-72px))] flex-col overflow-hidden rounded-2xl sm:left-[60px]",
          "border border-loma-line bg-loma-surface shadow-[0_30px_80px_-20px_rgba(0,0,0,0.9)]",
          "transition-all duration-200 ease-out",
          open
            ? "translate-x-0 opacity-100"
            : "pointer-events-none -translate-x-3 opacity-0",
        )}
      >
        <div className="flex items-center gap-2 px-3 pb-1.5 pt-3">
          <p className="text-[12.5px] font-medium text-loma-ink">Chats</p>
          <span className="font-mono text-[10px] text-loma-faint">
            {conversations.length}
          </span>
          <Button
            variant="ghost"
            size="icon-sm"
            className="ml-auto size-8 text-loma-faint hover:bg-white/[0.06] hover:text-loma-ink"
            onClick={onClose}
            aria-label="Close chats"
          >
            <X className="size-3.5" />
          </Button>
        </div>

        <div className="px-3 pb-2">
          <div className="group flex items-center gap-2 rounded-lg border border-transparent bg-white/[0.05] px-2.5 py-1.5 transition focus-within:border-loma-accent/30 focus-within:bg-white/[0.07]">
            <Search className="size-3.5 shrink-0 text-loma-faint" />
            <input
              ref={searchRef}
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search chats"
              aria-label="Search chats"
              className="w-full bg-transparent text-[12.5px] text-loma-ink outline-none placeholder:text-loma-faint"
            />
            {query ? (
              <button
                type="button"
                onClick={() => setQuery("")}
                aria-label="Clear search"
                className="grid min-h-[32px] min-w-[32px] place-items-center text-loma-faint transition hover:text-loma-ink"
              >
                <X className="size-3" />
              </button>
            ) : (
              <kbd className="shrink-0 rounded border border-loma-line px-1 font-mono text-[9px] text-loma-faint">
                ⌘K
              </kbd>
            )}
          </div>
        </div>

        <div className="px-3 pb-2">
          <Button
            onClick={onNew}
            variant="ghost"
            className="h-8 w-full justify-start gap-2 rounded-lg border border-loma-line-soft bg-white/[0.03] text-[12.5px] font-normal text-loma-ink hover:bg-white/[0.07]"
          >
            <Plus className="size-3.5 text-loma-accent" />
            New chat
          </Button>
        </div>

        <nav className="min-h-0 flex-1 overflow-y-auto px-2 pb-3">
          {loading && (
            <p className="px-3 py-8 text-center text-[12px] text-loma-faint">
              loading chats…
            </p>
          )}
          {error && (
            <div className="px-3 py-8 text-center">
              <p className="text-[12px] text-loma-dim">could not load chats</p>
              {onRetry && (
                <button
                  type="button"
                  onClick={onRetry}
                  className="mt-3 rounded-md border border-loma-line px-3 py-1.5 text-[11px] text-loma-ink hover:bg-white/[0.06]"
                >
                  Retry
                </button>
              )}
            </div>
          )}
          {!loading && !error && groups.length === 0 && (
            <div className="px-3 py-8 text-center">
              {query ? (
                <p className="text-[12px] text-loma-faint">
                  Nothing matches “{query}”.
                </p>
              ) : (
                <>
                  <p className="text-[12px] text-loma-faint">No chats yet.</p>
                  <button
                    type="button"
                    onClick={onNew}
                    className="mt-3 rounded-md border border-loma-line px-3 py-1.5 text-[11px] text-loma-ink hover:bg-white/[0.06]"
                  >
                    Start a chat
                  </button>
                </>
              )}
            </div>
          )}

          {groups.map((group) => (
            <div key={group.label} className="mb-3">
              <p className="px-2.5 pb-1 pt-2 font-mono text-[9.5px] uppercase tracking-[0.16em] text-loma-faint">
                {group.label}
              </p>
              <ul className="space-y-0.5">
                {group.items.map((conversation) => {
                  const active = conversation.id === activeId
                  const isRenaming = renaming === conversation.id
                  const isConfirming = confirming === conversation.id

                  return (
                    <li key={conversation.id}>
                      <div
                        className={cn(
                          "group/row relative flex items-center gap-1.5 rounded-lg pl-2.5 pr-1 text-[12.5px] transition",
                          active
                            ? "bg-white/[0.08] text-loma-ink"
                            : "text-loma-dim hover:bg-white/[0.045] hover:text-loma-ink",
                        )}
                      >
                        {active && (
                          <span className="absolute left-0 top-1/2 h-3.5 w-[2px] -translate-y-1/2 rounded-full bg-loma-accent" />
                        )}

                        {isRenaming ? (
                          <input
                            ref={renameRef}
                            value={draft}
                            onChange={(event) => setDraft(event.target.value)}
                            onKeyDown={(event) => {
                              if (event.key === "Enter") {
                                cancelledRef.current = false
                                commitRename(conversation.id)
                              }
                              if (event.key === "Escape") {
                                cancelledRef.current = true
                                setRenaming(null)
                              }
                            }}
                            onBlur={() => commitRename(conversation.id)}
                            aria-label="Chat title"
                            className="my-1 w-full rounded border border-loma-accent/40 bg-black/40 px-1.5 py-0.5 text-[12.5px] text-loma-ink outline-none"
                          />
                        ) : (
                          <button
                            onClick={() => onSelect(conversation.id)}
                            onDoubleClick={() => {
                              setRenaming(conversation.id)
                              setDraft(conversation.title)
                            }}
                            className="flex min-w-0 flex-1 items-center gap-2 py-[7px] text-left"
                            title={conversation.title}
                          >
                            <MessageSquare
                              className={cn(
                                "size-3 shrink-0 transition",
                                active ? "text-loma-accent" : "text-loma-faint/70",
                              )}
                            />
                            <span className="truncate">{conversation.title}</span>
                          </button>
                        )}

                        {isConfirming ? (
                          <span className="flex shrink-0 items-center gap-0.5">
                            <button
                              type="button"
                              onClick={() => onDelete(conversation.id)}
                              aria-label={`Confirm delete ${conversation.title}`}
                              className="grid min-h-[32px] min-w-[32px] place-items-center rounded text-loma-danger hover:bg-loma-danger/10 focus-visible:bg-loma-danger/10"
                            >
                              <Trash2 className="size-3.5" />
                            </button>
                            <button
                              type="button"
                              onClick={() => setConfirming(null)}
                              aria-label="Keep chat"
                              className="grid min-h-[32px] min-w-[32px] place-items-center rounded text-loma-faint hover:bg-white/[0.06] hover:text-loma-ink focus-visible:bg-white/[0.06]"
                            >
                              <Check className="size-3.5" />
                            </button>
                          </span>
                        ) : (
                          !isRenaming && (
                            <span
                              className={cn(
                                "flex shrink-0 items-center gap-0.5 transition sm:opacity-0 sm:group-hover/row:opacity-100 sm:focus-within:opacity-100",
                                active && "sm:opacity-100",
                              )}
                            >
                              <button
                                type="button"
                                onClick={() => {
                                  setRenaming(conversation.id)
                                  setDraft(conversation.title)
                                }}
                                aria-label={`Rename ${conversation.title}`}
                                className="grid min-h-[32px] min-w-[32px] place-items-center rounded text-loma-faint hover:bg-white/[0.06] hover:text-loma-ink focus-visible:bg-white/[0.06]"
                              >
                                <Pencil className="size-3" />
                              </button>
                              <button
                                type="button"
                                onClick={() => setConfirming(conversation.id)}
                                aria-label={`Delete ${conversation.title}`}
                                className="grid min-h-[32px] min-w-[32px] place-items-center rounded text-loma-faint hover:bg-white/[0.06] hover:text-loma-danger focus-visible:bg-white/[0.06]"
                              >
                                <Trash2 className="size-3" />
                              </button>
                            </span>
                          )
                        )}
                      </div>
                    </li>
                  )
                })}
              </ul>
            </div>
          ))}
        </nav>

        <div className="border-t border-loma-line-soft px-3 py-2">
          <p className="font-mono text-[9.5px] text-loma-faint/70">
            every chat is a json file on this machine
          </p>
        </div>
      </aside>
    </>
  )
}
