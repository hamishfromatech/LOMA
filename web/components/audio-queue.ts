"use client"

import { useCallback, useEffect, useRef } from "react"

/**
 * Plays the spoken sentences LOMA streams back, in order.
 *
 * Each `audio` event is a complete WAV, so they can be queued as data URLs and
 * played one after another. The browser's autoplay policy needs a user gesture
 * before the first sound — `unlock()` handles that on send.
 */
export function useAudioQueue() {
  const queue = useRef<string[]>([])
  const playing = useRef(false)
  const audio = useRef<HTMLAudioElement | null>(null)

  const ensureAudio = useCallback(() => {
    if (!audio.current) {
      const element = new Audio()
      element.preload = "auto"
      audio.current = element
    }
    return audio.current
  }, [])

  // The callback refers to itself, so keep it in a ref rather than recursing
  // through a `const` that is still being defined.
  const playNextRef = useRef<() => void>(() => undefined)

  const playNext = useCallback(() => {
    if (playing.current) return
    const next = queue.current.shift()
    if (!next) return

    const element = ensureAudio()
    playing.current = true
    element.src = next

    const finish = () => {
      playing.current = false
      playNextRef.current()
    }
    element.onended = finish
    element.onerror = finish
    void element.play().catch(finish)
  }, [ensureAudio])

  playNextRef.current = playNext

  const enqueue = useCallback(
    (base64: string) => {
      queue.current.push(`data:audio/wav;base64,${base64}`)
      playNext()
    },
    [playNext],
  )

  /** Called from a real click so later playback is allowed. */
  const unlock = useCallback(() => {
    const element = ensureAudio()
    element.muted = true
    void element
      .play()
      .catch(() => undefined)
      .finally(() => {
        element.muted = false
      })
  }, [ensureAudio])

  const stop = useCallback(() => {
    queue.current = []
    playing.current = false
    if (audio.current) {
      audio.current.pause()
      audio.current.removeAttribute("src")
    }
  }, [])

  useEffect(() => stop, [stop])

  return { enqueue, unlock, stop }
}
