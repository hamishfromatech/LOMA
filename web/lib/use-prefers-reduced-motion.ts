"use client"

import { useSyncExternalStore } from "react"

const QUERY = "(prefers-reduced-motion: reduce)"

function subscribe(callback: () => void) {
  const media = window.matchMedia(QUERY)
  media.addEventListener("change", callback)
  return () => media.removeEventListener("change", callback)
}

/**
 * true when the user asked the OS for reduced motion. CSS transitions are
 * separately disabled by the globals.css block; this is for JS-driven
 * animations (use-stick-to-bottom's resize, requestAnimationFrame tweens)
 * that CSS media queries cannot reach.
 */
export function usePrefersReducedMotion(): boolean {
  return useSyncExternalStore(
    subscribe,
    () => window.matchMedia(QUERY).matches,
    () => false,
  )
}
