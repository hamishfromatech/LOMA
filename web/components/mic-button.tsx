"use client"

import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { Loader2, Mic, Square } from "lucide-react"
import { useCallback, useEffect, useRef, useState } from "react"

type Props = {
  onTranscript: (text: string) => void
  disabled?: boolean
  className?: string
}

/**
 * Click to record, click again to stop.
 *
 * The clip goes to the Flask API, where faster-whisper transcribes it and the
 * text comes back to the composer.
 */
export function MicButton({ onTranscript, disabled, className }: Props) {
  const [recording, setRecording] = useState(false)
  const [working, setWorking] = useState(false)
  const [seconds, setSeconds] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const recorder = useRef<MediaRecorder | null>(null)
  const chunks = useRef<Blob[]>([])
  const stream = useRef<MediaStream | null>(null)

  useEffect(() => {
    if (!recording) return
    const timer = window.setInterval(() => setSeconds((s) => s + 1), 1000)
    return () => window.clearInterval(timer)
  }, [recording])

  const stopStream = useCallback(() => {
    stream.current?.getTracks().forEach((track) => track.stop())
    stream.current = null
  }, [])

  const start = useCallback(async () => {
    setError(null)
    try {
      const media = await navigator.mediaDevices.getUserMedia({ audio: true })
      stream.current = media
      chunks.current = []

      const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "audio/webm"
      const rec = new MediaRecorder(media, { mimeType: mime })
      recorder.current = rec

      rec.ondataavailable = (event) => {
        if (event.data.size > 0) chunks.current.push(event.data)
      }
      rec.onstop = async () => {
        stopStream()
        const blob = new Blob(chunks.current, { type: mime })
        setRecording(false)
        if (blob.size < 1000) return
        setWorking(true)
        try {
          const { transcribe } = await import("@/lib/api")
          const text = await transcribe(blob)
          if (text.trim()) onTranscript(text.trim())
          else setError("heard nothing")
        } catch {
          setError("transcription failed")
        } finally {
          setWorking(false)
        }
      }

      rec.start()
      setSeconds(0)
      setRecording(true)
    } catch {
      setError("microphone blocked")
      stopStream()
    }
  }, [onTranscript, stopStream])

  const stop = useCallback(() => {
    recorder.current?.stop()
    recorder.current = null
  }, [])

  useEffect(() => () => {
    recorder.current?.stop()
    stopStream()
  }, [stopStream])

  return (
    <div className={cn("flex items-center gap-2", className)}>
      <Button
        type="button"
        size="icon"
        variant={recording ? "destructive" : "ghost"}
        onClick={recording ? stop : start}
        disabled={disabled || working}
        aria-label={recording ? "Stop recording" : "Record a question"}
        className={cn(recording && "animate-pulse")}
      >
        {working ? (
          <Loader2 className="size-4 animate-spin" />
        ) : recording ? (
          <Square className="size-4" />
        ) : (
          <Mic className="size-4" />
        )}
      </Button>
      {recording && (
        <span className="font-mono text-xs text-red-400">
          rec {seconds}s
        </span>
      )}
      {error && <span className="text-xs text-red-400">{error}</span>}
    </div>
  )
}
