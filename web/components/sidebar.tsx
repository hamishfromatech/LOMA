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
}: Props) {
  const [query, setQuery] = useState("")
  const [renaming, setRenaming] = useState<string | null>(null)
  const [draft, setDraft] = useState("")
  const [confirming, setConfirming] = useState<string | null>(null)
  const searchRef = useRef<HTMLInputElement>(null)
  const renameRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (!open) {
      setRenaming(null)
      setConfirming(null)
      setQuery("")
      return
    }
    const timer = window.setTimeout(() => searchRef.current?.focus(), 180)
    return () => window.clearTimeout(timer)
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
        inert={!open}
        aria-hidden={!open}
        aria-label="Chats"
        className={cn(
          "fixed inset-y-3 left-[56px] z-40 flex w-[min(300px,calc(100vw-72px))] flex-col overflow-hidden rounded-2xl sm:left-[68px]",
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
            className="ml-auto size-7 text-loma-faint hover:bg-white/[0.06] hover:text-loma-ink"
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
                onClick={() => setQuery("")}
                aria-label="Clear search"
                className="text-loma-faint transition hover:text-loma-ink"
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
          {groups.length === 0 && (
            <p className="px-3 py-8 text-center text-[12px] text-loma-faint">
              {query ? `Nothing matches “${query}”.` : "No chats yet."}
            </p>
          )}

          {groups.map((group) => (
            <div key={group.label} className="mb-3">
              <p className="px-2.5 pb-1 pt-2 font-mono text-[9.5px] uppercase tracking-[0.16em] text-loma-faint/80">
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
                              if (event.key === "Enter") commitRename(conversation.id)
                              if (event.key === "Escape") setRenaming(null)
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
                              onClick={() => onDelete(conversation.id)}
                              aria-label={`Confirm delete ${conversation.title}`}
                              className="rounded p-1 text-loma-danger hover:bg-loma-danger/10"
                            >
                              <Check className="size-3.5" />
                            </button>
                            <button
                              onClick={() => setConfirming(null)}
                              aria-label="Keep chat"
                              className="rounded p-1 text-loma-faint hover:bg-white/[0.06] hover:text-loma-ink"
                            >
                              <X className="size-3.5" />
                            </button>
                          </span>
                        ) : (
                          !isRenaming && (
                            <span
                              className={cn(
                                "flex shrink-0 items-center gap-0.5 transition sm:opacity-0 sm:group-hover/row:opacity-100",
                                active && "sm:opacity-100",
                              )}
                            >
                              <button
                                onClick={() => {
                                  setRenaming(conversation.id)
                                  setDraft(conversation.title)
                                }}
                                aria-label={`Rename ${conversation.title}`}
                                className="rounded p-1 text-loma-faint hover:bg-white/[0.06] hover:text-loma-ink"
                              >
                                <Pencil className="size-3" />
                              </button>
                              <button
                                onClick={() => setConfirming(conversation.id)}
                                aria-label={`Delete ${conversation.title}`}
                                className="rounded p-1 text-loma-faint hover:bg-white/[0.06] hover:text-loma-danger"
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
