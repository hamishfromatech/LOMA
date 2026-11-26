"use client"

import {
  createContext,
  useCallback,
  useContext,
  useRef,
  useState,
  type ReactNode,
} from "react"
import { TriangleAlert, X } from "lucide-react"

type Toast = {
  id: string
  title: string
  description?: string
}

const ToastContext = createContext<(toast: Omit<Toast, "id">) => void>(() => {})

let nextId = 0

/**
 * Stack of transient, non-blocking notifications for action failures that
 * should not hijack the conversation (voice toggle, refresh, preview).
 * Each toast lives ~4.5s and is announced via aria-live.
 */
export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([])
  const timers = useRef<Map<string, number>>(new Map())

  const dismiss = useCallback((id: string) => {
    setToasts((prev) => prev.filter((toast) => toast.id !== id))
    const timer = timers.current.get(id)
    if (timer !== undefined) {
      window.clearTimeout(timer)
      timers.current.delete(id)
    }
  }, [])

  const push = useCallback((toast: Omit<Toast, "id">) => {
    const id = `toast-${nextId++}`
    setToasts((prev) => [...prev.slice(-2), { id, ...toast }])
    const timer = window.setTimeout(() => dismiss(id), 4500)
    timers.current.set(id, timer)
  }, [dismiss])

  return (
    <ToastContext.Provider value={push}>
      {children}

      <div
        aria-live="assertive"
        role="status"
        className="pointer-events-none fixed bottom-4 right-4 z-[80] flex w-[min(340px,calc(100vw-32px))] flex-col gap-2"
      >
        {toasts.map((toast) => (
          <div
            key={toast.id}
            role="alert"
            className="pointer-events-auto flex items-start gap-2.5 rounded-xl border border-loma-line bg-loma-raise px-3.5 py-3 shadow-[0_20px_50px_-20px_rgba(0,0,0,0.85)]"
          >
            <TriangleAlert className="mt-0.5 size-4 shrink-0 text-loma-danger" aria-hidden />
            <div className="min-w-0 flex-1">
              <p className="text-[12.5px] font-medium leading-snug text-loma-ink">
                {toast.title}
              </p>
              {toast.description && (
                <p className="mt-0.5 text-[11.5px] leading-snug text-loma-dim">
                  {toast.description}
                </p>
              )}
            </div>
            <button
              type="button"
              onClick={() => dismiss(toast.id)}
              aria-label="Dismiss notification"
              className="grid size-7 shrink-0 place-items-center rounded-md text-loma-faint transition hover:bg-white/[0.06] hover:text-loma-ink"
            >
              <X className="size-3.5" />
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  )
}

export const useToast = () => useContext(ToastContext)
