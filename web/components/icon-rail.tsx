"use client"

import { LomaMark } from "@/components/brand"
import { cn } from "@/lib/utils"
import { PanelLeft, Settings2, SquarePen } from "lucide-react"
import Link from "next/link"

type Props = {
  onNew: () => void
  onToggleChats: () => void
  chatsOpen: boolean
  className?: string
}

function RailButton({
  label,
  active,
  onClick,
  children,
}: {
  label: string
  active?: boolean
  onClick?: () => void
  children: React.ReactNode
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      title={label}
      aria-pressed={active}
      className={cn(
        "grid size-9 place-items-center rounded-[10px] transition-colors duration-150",
        active
          ? "bg-white/[0.09] text-loma-ink"
          : "text-loma-faint hover:bg-white/[0.06] hover:text-loma-ink",
      )}
    >
      {children}
    </button>
  )
}

/** The narrow rail on the far left — mark, new chat, chats, settings. */
export function IconRail({ onNew, onToggleChats, chatsOpen, className }: Props) {
  return (
    <div
      className={cn(
        "flex w-12 shrink-0 flex-col items-center gap-1 py-3 sm:w-[60px]",
        className,
      )}
    >
      <Link href="/" aria-label="LOMA home" className="mb-4">
        <LomaMark size={30} />
      </Link>

      <RailButton label="New chat" onClick={onNew}>
        <SquarePen className="size-[17px]" strokeWidth={1.8} />
      </RailButton>

      <RailButton label="Chats (⌘K)" active={chatsOpen} onClick={onToggleChats}>
        <PanelLeft className="size-[17px]" strokeWidth={1.8} />
      </RailButton>

      <div className="mt-auto flex flex-col items-center gap-1">
        <span className="mb-2 h-px w-5 bg-loma-line" />
        <Link
          href="/settings"
          aria-label="Settings"
          title="Settings"
          className="grid size-9 place-items-center rounded-[10px] text-loma-faint transition-colors hover:bg-white/[0.06] hover:text-loma-ink"
        >
          <Settings2 className="size-[17px]" strokeWidth={1.8} />
        </Link>
        <div
          className="grid size-7 place-items-center rounded-full border border-loma-line bg-white/[0.05] text-[10px] font-medium text-loma-dim"
          title="Local account"
        >
          H
        </div>
      </div>
    </div>
  )
}
