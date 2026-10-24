"use client"

import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { PanelLeft, Settings2, SquarePen, Waves } from "lucide-react"
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
    <Button
      variant="ghost"
      size="icon-lg"
      onClick={onClick}
      aria-label={label}
      title={label}
      className={cn(
        "size-9 rounded-lg text-white/55 hover:bg-white/[0.07] hover:text-white",
        active && "bg-white/[0.08] text-white",
      )}
    >
      {children}
    </Button>
  )
}

/** The narrow icon rail on the far left — Open WebUI's silhouette. */
export function IconRail({ onNew, onToggleChats, chatsOpen, className }: Props) {
  return (
    <div
      className={cn(
        "flex w-14 shrink-0 flex-col items-center gap-1.5 bg-transparent py-3",
        className,
      )}
    >
      <div className="mb-3 grid size-8 place-items-center rounded-lg bg-gradient-to-br from-amber-300 to-orange-500 text-black">
        <Waves className="size-4" />
      </div>

      <RailButton label="New chat" onClick={onNew}>
        <SquarePen className="size-[18px]" />
      </RailButton>

      <RailButton label="Chats" active={chatsOpen} onClick={onToggleChats}>
        <PanelLeft className="size-[18px]" />
      </RailButton>

      <div className="mt-auto flex flex-col items-center gap-1.5">
        <Link
          href="/settings"
          aria-label="Settings"
          title="Settings"
          className="grid size-9 place-items-center rounded-lg text-white/55 transition hover:bg-white/[0.07] hover:text-white"
        >
          <Settings2 className="size-[18px]" />
        </Link>
        <div className="mt-1 grid size-8 place-items-center rounded-full bg-white/10 text-[11px] font-medium text-white/70">
          H
        </div>
      </div>
    </div>
  )
}
