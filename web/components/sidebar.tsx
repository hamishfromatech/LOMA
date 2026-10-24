"use client"

import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import type { ConversationSummary } from "@/lib/types"
import { MessageSquare, Plus, Search, Trash2, X } from "lucide-react"
import { useMemo, useState } from "react"

type Props = {
  conversations: ConversationSummary[]
  activeId: string | null
  open: boolean
  onClose: () => void
  onNew: () => void
  onSelect: (id: string) => void
  onDelete: (id: string) => void
}

function dayKey(iso: string): string {
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

/** The conversation drawer, opened from the rail. */
export function ChatsDrawer({
  conversations,
  activeId,
  open,
  onClose,
  onNew,
  onSelect,
  onDelete,
}: Props) {
  const [query, setQuery] = useState("")

  const groups = useMemo(() => {
    const filtered = query
      ? conversations.filter((c) =>
          c.title.toLowerCase().includes(query.toLowerCase()),
        )
      : conversations

    const buckets = new Map<string, ConversationSummary[]>()
    for (const conversation of filtered) {
      const key = dayKey(conversation.updated_at)
      const list = buckets.get(key) ?? []
      list.push(conversation)
      buckets.set(key, list)
    }

    return ["Today", "Yesterday", "Previous 7 days", "Previous 30 days", "Older"]
      .filter((key) => buckets.has(key))
      .map((key) => ({ label: key, items: buckets.get(key)! }))
  }, [conversations, query])

  return (
    <>
      {open && (
        <div
          className="fixed inset-0 z-30 bg-black/50 backdrop-blur-[2px]"
          onClick={onClose}
          aria-hidden
        />
      )}

      <aside
        inert={!open}
        aria-hidden={!open}
        className={cn(
          "fixed inset-y-2 left-16 z-40 flex w-[290px] flex-col overflow-hidden rounded-xl border border-white/[0.08] bg-[#1c1c1c] shadow-2xl shadow-black/60 transition-all duration-200",
          open
            ? "translate-x-0 opacity-100"
            : "pointer-events-none -translate-x-3 opacity-0",
        )}
      >
        <div className="flex items-center gap-2 px-3.5 pb-2 pt-3.5">
          <p className="text-[13px] font-medium text-white/90">Chats</p>
          <Button
            variant="ghost"
            size="icon-sm"
            className="ml-auto text-white/40 hover:text-white"
            onClick={onClose}
            aria-label="Close chats"
          >
            <X className="size-4" />
          </Button>
        </div>

        <div className="px-3 pb-2">
          <div className="flex items-center gap-2 rounded-lg bg-white/[0.05] px-2.5 py-1.5">
            <Search className="size-3.5 text-white/35" />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search chats"
              aria-label="Search chats"
              className="w-full bg-transparent text-[13px] text-white/80 outline-none placeholder:text-white/30"
            />
          </div>
        </div>

        <div className="px-3 pb-2">
          <Button
            onClick={onNew}
            variant="ghost"
            className="w-full justify-start gap-2 rounded-lg bg-white/[0.05] text-[13px] text-white/80 hover:bg-white/[0.09]"
          >
            <Plus className="size-4" />
            New chat
          </Button>
        </div>

        <nav className="min-h-0 flex-1 overflow-y-auto px-2 pb-3">
          {groups.length === 0 && (
            <p className="px-3 py-6 text-xs text-white/25">No chats yet.</p>
          )}

          {groups.map((group) => (
            <div key={group.label} className="mb-2">
              <p className="px-2.5 pb-1 pt-2 font-mono text-[10px] uppercase tracking-wider text-white/25">
                {group.label}
              </p>
              <ul>
                {group.items.map((conversation) => (
                  <li key={conversation.id}>
                    <div
                      className={cn(
                        "group flex items-center gap-2 rounded-lg px-2.5 py-[7px] text-[13px] transition",
                        conversation.id === activeId
                          ? "bg-white/[0.09] text-white"
                          : "text-white/65 hover:bg-white/[0.05] hover:text-white/90",
                      )}
                    >
                      <button
                        onClick={() => onSelect(conversation.id)}
                        className="flex min-w-0 flex-1 items-center gap-2 text-left"
                      >
                        <MessageSquare className="size-3.5 shrink-0 text-white/25" />
                        <span className="truncate">{conversation.title}</span>
                      </button>
                      <button
                        onClick={() => onDelete(conversation.id)}
                        aria-label={`Delete ${conversation.title}`}
                        className="shrink-0 rounded p-1 text-white/0 transition group-hover:text-white/30 hover:!text-red-400"
                      >
                        <Trash2 className="size-3.5" />
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </nav>
      </aside>
    </>
  )
}
