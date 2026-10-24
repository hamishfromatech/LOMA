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
    <div className={cn("flex items-center gap-1.5", className)}>
      {recording && (
        <span className="flex items-center gap-1.5 rounded-full bg-loma-danger/10 px-2 py-1 font-mono text-[10px] tabular-nums text-loma-danger">
          <span className="loma-breathe size-1.5 rounded-full bg-loma-danger" />
          {String(Math.floor(seconds / 60)).padStart(2, "0")}:
          {String(seconds % 60).padStart(2, "0")}
        </span>
      )}

      {error && !recording && (
        <span
          role="status"
          className="max-w-[130px] truncate font-mono text-[10px] text-loma-danger"
        >
          {error}
        </span>
      )}

      <Button
        type="button"
        size="icon"
        variant="ghost"
        onClick={recording ? stop : start}
        disabled={disabled || working}
        aria-label={recording ? "Stop recording" : "Record a question"}
        title={recording ? "Stop recording" : "Record a question"}
        className={cn(
          "size-8 rounded-full transition-colors",
          recording
            ? "bg-loma-danger/15 text-loma-danger hover:bg-loma-danger/25"
            : "text-loma-dim hover:bg-white/[0.06] hover:text-loma-ink",
        )}
      >
        {working ? (
          <Loader2 className="size-4 animate-spin" />
        ) : recording ? (
          <Square className="size-3 fill-current" />
        ) : (
          <Mic className="size-4" />
        )}
      </Button>
    </div>
  )
}
