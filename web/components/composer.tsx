"use client"

import { MicButton } from "@/components/mic-button"
import {
  FileUpload,
  FileUploadContent,
  FileUploadTrigger,
} from "@/components/ui/file-upload"
import { Button } from "@/components/ui/button"
import {
  PromptInput,
  PromptInputActions,
  PromptInputTextarea,
} from "@/components/ui/prompt-input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { cn } from "@/lib/utils"
import type { ModelOption, SettingsPayload } from "@/lib/types"
import {
  ArrowUp,
  ChevronDown,
  HardDrive,
  ImagePlus,
  RefreshCw,
  Square,
  Volume2,
  VolumeX,
  X,
} from "lucide-react"

type Props = {
  value: string
  onValueChange: (value: string) => void
  onSubmit: () => void
  busy: boolean
  onStop: () => void
  voiceOn: boolean
  onToggleVoice: () => void
  onTranscript: (text: string) => void
  settings: SettingsPayload | null
  models: ModelOption[]
  onModelChange: (model: string) => void
  onRefreshModels: () => void
  modelsLoading?: boolean
  variant?: "hero" | "docked"
  attachments: { id: string; dataUrl: string }[]
  onAddAttachments: (files: File[]) => void
  onRemoveAttachment: (id: string) => void
  visionEnabled: boolean
}

export function Composer({
  value,
  onValueChange,
  onSubmit,
  busy,
  onStop,
  voiceOn,
  onToggleVoice,
  onTranscript,
  settings,
  models,
  onModelChange,
  onRefreshModels,
  modelsLoading = false,
  variant = "docked",
  attachments,
  onAddAttachments,
  onRemoveAttachment,
  visionEnabled,
}: Props) {
  const ready = value.trim().length > 0 || attachments.length > 0

  return (
    <div className="w-full">
      <FileUpload
        onFilesAdded={onAddAttachments}
        accept="image/png,image/jpeg,image/webp,image/gif"
        multiple
        disabled={!visionEnabled}
      >
      <PromptInput
        value={value}
        onValueChange={onValueChange}
        onSubmit={onSubmit}
        isLoading={busy}
        maxHeight={220}
        className={cn(
          "rounded-[22px] border border-loma-line bg-loma-surface px-2.5 pb-2 pt-2.5",
          "shadow-[0_1px_0_rgba(255,250,242,0.045)_inset,0_20px_50px_-30px_rgba(0,0,0,0.95)]",
          "transition-colors duration-150 focus-within:border-loma-accent/35",
          variant === "hero" && "shadow-[0_1px_0_rgba(255,250,242,0.05)_inset,0_28px_70px_-34px_rgba(0,0,0,1)]",
        )}
      >
        {attachments.length > 0 && (
          <div className="flex flex-wrap gap-2 px-2 pb-1 pt-1">
            {attachments.map((attachment) => (
              <div
                key={attachment.id}
                className="group/thumb relative size-16 overflow-hidden rounded-lg border border-loma-line bg-black/40"
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={attachment.dataUrl}
                  alt="Attached image"
                  className="size-full object-cover"
                />
                <button
                  type="button"
                  onClick={() => onRemoveAttachment(attachment.id)}
                  aria-label="Remove image"
                  className="absolute right-1 top-1 grid size-5 place-items-center rounded-full bg-black/70 text-loma-ink opacity-0 transition group-hover/thumb:opacity-100 focus-visible:opacity-100"
                >
                  <X className="size-3" />
                </button>
              </div>
            ))}
          </div>
        )}

        <PromptInputTextarea
          id="loma-prompt"
          name="message"
          aria-label="Message LOMA"
          placeholder="Ask anything. Nothing leaves this machine."
          className={cn(
            "min-h-[46px] rounded-none border-0 bg-transparent px-2.5 py-1.5",
            "text-[15px] leading-relaxed text-loma-ink shadow-none",
            "placeholder:text-loma-faint/85",
            "focus-visible:border-0 focus-visible:ring-0 dark:bg-transparent",
          )}
        />

        <PromptInputActions className="justify-between px-0 pb-0 pt-0.5">
          <div className="flex items-center gap-1">
            <FileUploadTrigger
              aria-label={visionEnabled ? "Attach an image" : "This model cannot read images"}
              title={visionEnabled ? "Attach an image" : "This model cannot read images"}
              disabled={!visionEnabled}
              className={cn(
                "grid size-7 place-items-center rounded-full transition",
                visionEnabled
                  ? "text-loma-dim hover:bg-white/[0.06] hover:text-loma-ink"
                  : "cursor-not-allowed text-loma-faint/40",
              )}
            >
              <ImagePlus className="size-4" />
            </FileUploadTrigger>

            <button
              type="button"
              onClick={onToggleVoice}
              aria-pressed={voiceOn}
              aria-label={voiceOn ? "Spoken replies are on" : "Spoken replies are off"}
              className={cn(
                "flex h-7 items-center gap-1.5 rounded-full px-2 font-mono text-[10.5px] uppercase tracking-[0.1em] transition",
                voiceOn
                  ? "bg-loma-teal/10 text-loma-teal hover:bg-loma-teal/15"
                  : "text-loma-faint hover:bg-white/[0.06] hover:text-loma-dim",
              )}
            >
              {voiceOn ? <Volume2 className="size-3.5" /> : <VolumeX className="size-3.5" />}
              <span className="hidden sm:inline">{voiceOn ? "voice" : "muted"}</span>
            </button>
          </div>

          <div className="flex items-center gap-1">
            {settings && (
              <Select
                value={settings.llm_model}
                onValueChange={(next) => onModelChange(String(next))}
                onOpenChange={(open) => {
                  // Ollama's list changes whenever the user pulls a model.
                  if (open) onRefreshModels()
                }}
              >
                <SelectTrigger
                  size="sm"
                  aria-label="Language model"
                  title="Models installed in Ollama"
                  className={cn(
                    "h-7 max-w-[210px] gap-1 rounded-full border-0 bg-transparent px-2.5",
                    "font-mono text-[11px] text-loma-dim shadow-none",
                    "hover:bg-white/[0.06] hover:text-loma-ink",
                  )}
                >
                  <SelectValue />
                  <ChevronDown className="size-3 opacity-50" />
                </SelectTrigger>

                <SelectContent
                  alignItemWithTrigger={false}
                  sideOffset={8}
                  align="end"
                  className="w-[300px] border border-loma-line p-1 shadow-2xl shadow-black/70"
                >
                  {models.length === 0 && (
                    <div className="px-3 py-3 text-[11.5px] leading-relaxed text-loma-faint">
                      No chat models found in Ollama.
                      <br />
                      <span className="font-mono text-loma-dim">ollama pull qwen3.5:0.8b</span>
                    </div>
                  )}

                  {models.map((model) => (
                    <SelectItem key={model.name} value={model.name} className="py-1.5">
                      <span className="flex w-full items-center gap-2">
                        <span className="min-w-0 flex-1">
                          <span className="block truncate font-mono text-[12px] text-loma-ink">
                            {model.name}
                          </span>
                          {model.parameters || model.quantization ? (
                            <span className="block truncate font-mono text-[10px] text-loma-faint">
                              {[model.parameters, model.quantization, model.family]
                                .filter(Boolean)
                                .join(" · ")}
                            </span>
                          ) : null}
                        </span>
                        {model.size_mb > 0 && (
                          <span className="shrink-0 font-mono text-[10px] tabular-nums text-loma-faint">
                            {(model.size_mb / 1024).toFixed(1)} GB
                          </span>
                        )}
                      </span>
                    </SelectItem>
                  ))}

                  <div className="mt-1 flex items-center justify-between border-t border-loma-line-soft px-2 pt-1.5 pb-1">
                    <span className="flex items-center gap-1.5 font-mono text-[9.5px] uppercase tracking-[0.12em] text-loma-faint">
                      <HardDrive className="size-2.5" />
                      {models.length} installed
                    </span>
                    <button
                      type="button"
                      onClick={(event) => {
                        event.preventDefault()
                        event.stopPropagation()
                        onRefreshModels()
                      }}
                      className="flex items-center gap-1 rounded px-1 py-0.5 font-mono text-[9.5px] uppercase tracking-[0.12em] text-loma-faint transition hover:bg-white/[0.06] hover:text-loma-ink"
                    >
                      <RefreshCw className={cn("size-2.5", modelsLoading && "animate-spin")} />
                      rescan
                    </button>
                  </div>
                </SelectContent>
              </Select>
            )}

            <MicButton onTranscript={onTranscript} disabled={busy} />

            <Button
              type="button"
              size="icon"
              onClick={busy ? onStop : ready ? onSubmit : undefined}
              disabled={!busy && !ready}
              aria-label={busy ? "Stop generating" : "Send message"}
              title={busy ? "Stop" : "Send"}
              className={cn(
                "ml-0.5 size-8 rounded-full transition-all duration-150",
                busy
                  ? "bg-loma-ink text-[#161514] hover:bg-white"
                  : ready
                    ? "bg-loma-ink text-[#161514] hover:bg-white active:scale-95"
                    : "cursor-default bg-white/[0.07] text-loma-faint",
              )}
            >
              {busy ? <Square className="size-3 fill-current" /> : <ArrowUp className="size-4" />}
            </Button>
          </div>
        </PromptInputActions>
      </PromptInput>

      <FileUploadContent>
        <div className="pointer-events-none rounded-2xl border-2 border-dashed border-loma-accent/50 bg-loma-bg/70 px-10 py-8 text-center">
          <ImagePlus className="mx-auto mb-3 size-6 text-loma-accent" />
          <p className="text-[14px] text-loma-ink">Drop an image to show LOMA</p>
          <p className="mt-1 font-mono text-[10.5px] text-loma-faint">
            png · jpeg · webp · gif — up to 4 per turn
          </p>
        </div>
      </FileUploadContent>
      </FileUpload>

      <div className="mt-2 flex items-center justify-between px-2.5 font-mono text-[10px] text-loma-faint/75">
        <span className="hidden items-center gap-1.5 sm:flex">
          <kbd className="rounded border border-loma-line bg-white/[0.04] px-1 py-px text-[9px] text-loma-dim">
            ↵
          </kbd>
          send
          <span className="text-loma-faint/40">·</span>
          <kbd className="rounded border border-loma-line bg-white/[0.04] px-1 py-px text-[9px] text-loma-dim">
            ⇧↵
          </kbd>
          new line
        </span>
        <span className="truncate">
          {settings
            ? `${Math.round(settings.context.window / 1024)}k context · offline`
            : "offline"}
        </span>
      </div>
    </div>
  )
}
