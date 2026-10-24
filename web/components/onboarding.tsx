"use client"

import { LomaMark } from "@/components/brand"
import { cn } from "@/lib/utils"
import { ArrowLeft, ArrowRight, Check, SkipForward, Sparkles } from "lucide-react"
import { useCallback, useMemo, useRef, useState } from "react"

export type OnboardingAnswers = {
  name: string
  role: string
  goals: string[]
  style: string
  notes: string
}

const GOALS = [
  "writing",
  "coding",
  "research",
  "admin and email",
  "planning my day",
  "thinking out loud",
]

const STYLES = [
  { value: "short and direct", hint: "one or two sentences, no warm-up" },
  { value: "thorough", hint: "the reasoning, then the answer" },
  { value: "warm and conversational", hint: "the way a friend would explain it" },
]

type Step = {
  id: keyof OnboardingAnswers
  kicker: string
  title: string
  hint: string
}

const STEPS: Step[] = [
  {
    id: "name",
    kicker: "01 — who",
    title: "What should I call you?",
    hint: "It goes into the memory file on this machine, and nowhere else.",
  },
  {
    id: "role",
    kicker: "02 — what you do",
    title: "What are you working on these days?",
    hint: "A sentence is plenty. It helps me assume less.",
  },
  {
    id: "goals",
    kicker: "03 — why LOMA",
    title: "What do you mostly want help with?",
    hint: "Pick any that apply, or type your own.",
  },
  {
    id: "style",
    kicker: "04 — how I answer",
    title: "How do you like being answered?",
    hint: "Spoken aloud, so this matters more than it looks.",
  },
  {
    id: "notes",
    kicker: "05 — anything else",
    title: "Anything I should know to communicate well with you?",
    hint: "Pace, hearing, attention, energy — whatever is true. Optional.",
  },
]

const LAST = STEPS.length - 1

export function Onboarding({
  onComplete,
  onSkip,
}: {
  onComplete: (answers: OnboardingAnswers) => Promise<void> | void
  onSkip: () => Promise<void> | void
}) {
  const [step, setStep] = useState(0)
  const [answers, setAnswers] = useState<OnboardingAnswers>({
    name: "",
    role: "",
    goals: [],
    style: "",
    notes: "",
  })
  const [customGoal, setCustomGoal] = useState("")
  const [busy, setBusy] = useState(false)
  const [recap, setRecap] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const areaRef = useRef<HTMLTextAreaElement>(null)

  const current = STEPS[step]
  const isLast = step === LAST

  const canAdvance = useMemo(() => {
    if (current.id === "name") return answers.name.trim().length > 0
    if (current.id === "role") return answers.role.trim().length > 0
    return true
  }, [answers, current.id])

  const focusFirst = useCallback(() => {
    window.setTimeout(() => {
      inputRef.current?.focus()
      areaRef.current?.focus()
    }, 60)
  }, [])

  const goNext = useCallback(() => {
    if (!canAdvance) return
    if (isLast) {
      setRecap(true)
      return
    }
    setStep((value) => Math.min(LAST, value + 1))
    focusFirst()
  }, [canAdvance, focusFirst, isLast])

  const goBack = useCallback(() => {
    if (recap) {
      setRecap(false)
      return
    }
    setStep((value) => Math.max(0, value - 1))
  }, [recap])

  const toggleGoal = (goal: string) => {
    setAnswers((value) => ({
      ...value,
      goals: value.goals.includes(goal)
        ? value.goals.filter((item) => item !== goal)
        : [...value.goals, goal],
    }))
  }

  const addCustomGoal = () => {
    const goal = customGoal.trim()
    if (!goal) return
    setAnswers((value) =>
      value.goals.includes(goal) ? value : { ...value, goals: [...value.goals, goal] },
    )
    setCustomGoal("")
  }

  const finish = async () => {
    setBusy(true)
    try {
      await onComplete(answers)
    } finally {
      setBusy(false)
    }
  }

  const skip = async () => {
    setBusy(true)
    try {
      await onSkip()
    } finally {
      setBusy(false)
    }
  }

  // Enter advances; Escape is deliberately not bound — losing five answers to a
  // stray keypress is worse than having no shortcut.
  const onKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault()
      if (recap) void finish()
      else goNext()
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-loma-bg/95 px-4 backdrop-blur-sm">
      <div
        aria-hidden
        className="pointer-events-none absolute inset-x-0 top-0 h-[420px] bg-[radial-gradient(55%_100%_at_50%_0%,rgba(227,168,92,0.07),transparent_70%)]"
      />

      <div className="relative w-full max-w-[620px]" onKeyDown={onKeyDown}>
        <div className="mb-7 flex flex-col items-center text-center">
          <LomaMark size={44} className="mb-5" />
          {!recap ? (
            <>
              <p className="font-mono text-[10px] uppercase tracking-[0.22em] text-loma-accent/80">
                {current.kicker}
              </p>
              <h1 className="mt-3 text-[26px] font-medium leading-tight tracking-[-0.02em] text-loma-ink">
                {current.title}
              </h1>
              <p className="mt-2 max-w-[420px] text-[13px] leading-relaxed text-loma-faint">
                {current.hint}
              </p>
            </>
          ) : (
            <>
              <p className="font-mono text-[10px] uppercase tracking-[0.22em] text-loma-accent/80">
                06 — check
              </p>
              <h1 className="mt-3 text-[26px] font-medium leading-tight tracking-[-0.02em] text-loma-ink">
                Here is what I will remember.
              </h1>
              <p className="mt-2 text-[13px] leading-relaxed text-loma-faint">
                Written to <span className="font-mono text-loma-dim">MEMORY.md</span> on this
                machine. Editable at any time in Settings.
              </p>
            </>
          )}
        </div>

        <div className="rounded-[22px] border border-loma-line bg-loma-raise p-5 shadow-[0_1px_0_rgba(255,250,242,0.05)_inset,0_28px_70px_-30px_rgba(0,0,0,1)]">
          {recap ? (
            <ul className="space-y-2.5">
              {[
                ["Name", answers.name],
                ["Working on", answers.role],
                ["Wants help with", answers.goals.join(", ")],
                ["Answers", answers.style],
                ["Context", answers.notes],
              ].map(([label, value]) => (
                <li
                  key={label}
                  className={cn(
                    "flex gap-3 border-b border-loma-line-soft pb-2.5 last:border-0 last:pb-0",
                    !value && "opacity-35",
                  )}
                >
                  <span className="w-[124px] shrink-0 font-mono text-[10px] uppercase tracking-[0.12em] text-loma-faint">
                    {label}
                  </span>
                  <span className="min-w-0 flex-1 text-[13.5px] text-loma-ink">
                    {value || "—"}
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <>
              {current.id === "name" && (
                <input
                  ref={inputRef}
                  value={answers.name}
                  onChange={(event) =>
                    setAnswers((value) => ({ ...value, name: event.target.value }))
                  }
                  placeholder="Your first name"
                  aria-label="Your name"
                  autoComplete="given-name"
                  className="w-full rounded-xl border border-loma-line bg-black/25 px-3.5 py-3 text-[15px] text-loma-ink outline-none transition placeholder:text-loma-faint/60 focus:border-loma-accent/40"
                />
              )}

              {current.id === "role" && (
                <textarea
                  ref={areaRef}
                  value={answers.role}
                  onChange={(event) =>
                    setAnswers((value) => ({ ...value, role: event.target.value }))
                  }
                  rows={3}
                  placeholder="I run a small YouTube channel and write about hardware."
                  aria-label="What you are working on"
                  className="w-full resize-none rounded-xl border border-loma-line bg-black/25 px-3.5 py-3 text-[15px] leading-relaxed text-loma-ink outline-none transition placeholder:text-loma-faint/60 focus:border-loma-accent/40"
                />
              )}

              {current.id === "goals" && (
                <div className="space-y-3">
                  <div className="flex flex-wrap gap-2">
                    {[...GOALS, ...answers.goals.filter((g) => !GOALS.includes(g))].map(
                      (goal) => {
                        const on = answers.goals.includes(goal)
                        return (
                          <button
                            key={goal}
                            type="button"
                            onClick={() => toggleGoal(goal)}
                            aria-pressed={on}
                            className={cn(
                              "rounded-full border px-3 py-1.5 text-[12.5px] transition",
                              on
                                ? "border-loma-accent/40 bg-loma-accent/10 text-loma-accent"
                                : "border-loma-line text-loma-dim hover:border-loma-line hover:bg-white/[0.04] hover:text-loma-ink",
                            )}
                          >
                            {on && <Check className="mr-1 inline size-3 align-[-1px]" />}
                            {goal}
                          </button>
                        )
                      },
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    <input
                      value={customGoal}
                      onChange={(event) => setCustomGoal(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter") {
                          event.preventDefault()
                          event.stopPropagation()
                          addCustomGoal()
                        }
                      }}
                      placeholder="Something else…"
                      aria-label="Add your own goal"
                      className="flex-1 rounded-xl border border-loma-line bg-black/25 px-3 py-2 text-[13px] text-loma-ink outline-none placeholder:text-loma-faint/60 focus:border-loma-accent/40"
                    />
                    <button
                      type="button"
                      onClick={addCustomGoal}
                      className="rounded-xl border border-loma-line px-3 py-2 font-mono text-[10px] uppercase tracking-[0.12em] text-loma-dim transition hover:bg-white/[0.05] hover:text-loma-ink"
                    >
                      add
                    </button>
                  </div>
                </div>
              )}

              {current.id === "style" && (
                <div className="space-y-2">
                  {STYLES.map((option) => {
                    const on = answers.style === option.value
                    return (
                      <button
                        key={option.value}
                        type="button"
                        onClick={() =>
                          setAnswers((value) => ({ ...value, style: option.value }))
                        }
                        aria-pressed={on}
                        className={cn(
                          "flex w-full items-center gap-3 rounded-xl border px-3.5 py-2.5 text-left transition",
                          on
                            ? "border-loma-accent/40 bg-loma-accent/[0.07]"
                            : "border-loma-line hover:bg-white/[0.04]",
                        )}
                      >
                        <span
                          className={cn(
                            "grid size-4 shrink-0 place-items-center rounded-full border",
                            on
                              ? "border-loma-accent bg-loma-accent text-[#1b1713]"
                              : "border-loma-line text-transparent",
                          )}
                        >
                          <Check className="size-2.5" />
                        </span>
                        <span className="min-w-0">
                          <span className="block text-[13.5px] text-loma-ink">
                            {option.value}
                          </span>
                          <span className="block font-mono text-[10.5px] text-loma-faint">
                            {option.hint}
                          </span>
                        </span>
                      </button>
                    )
                  })}
                </div>
              )}

              {current.id === "notes" && (
                <textarea
                  ref={areaRef}
                  value={answers.notes}
                  onChange={(event) =>
                    setAnswers((value) => ({ ...value, notes: event.target.value }))
                  }
                  rows={3}
                  placeholder="I read more than I listen, so written answers help. Keep it slow."
                  aria-label="Anything else LOMA should know"
                  className="w-full resize-none rounded-xl border border-loma-line bg-black/25 px-3.5 py-3 text-[15px] leading-relaxed text-loma-ink outline-none transition placeholder:text-loma-faint/60 focus:border-loma-accent/40"
                />
              )}
            </>
          )}
        </div>

        <div className="mt-4 flex items-center gap-2">
          <div className="flex flex-1 items-center gap-1.5">
            {STEPS.map((item, index) => (
              <span
                key={item.id}
                className={cn(
                  "h-1 rounded-full transition-all duration-300",
                  recap || index < step
                    ? "w-6 bg-loma-accent/60"
                    : index === step
                      ? "w-6 bg-loma-accent"
                      : "w-3 bg-white/10",
                )}
              />
            ))}
          </div>

          {!recap && (
            <button
              type="button"
              onClick={() => void skip()}
              disabled={busy}
              className="flex items-center gap-1.5 rounded-lg px-2.5 py-2 font-mono text-[10px] uppercase tracking-[0.12em] text-loma-faint transition hover:bg-white/[0.05] hover:text-loma-dim"
            >
              <SkipForward className="size-3" />
              skip
            </button>
          )}

          {(step > 0 || recap) && (
            <button
              type="button"
              onClick={goBack}
              disabled={busy}
              className="flex items-center gap-1.5 rounded-lg border border-loma-line px-3 py-2 text-[12.5px] text-loma-dim transition hover:bg-white/[0.05] hover:text-loma-ink"
            >
              <ArrowLeft className="size-3.5" />
              back
            </button>
          )}

          <button
            type="button"
            onClick={() => (recap ? void finish() : goNext())}
            disabled={!canAdvance || busy}
            className={cn(
              "flex items-center gap-1.5 rounded-lg px-3.5 py-2 text-[12.5px] font-medium transition",
              canAdvance && !busy
                ? "bg-loma-ink text-[#161514] hover:bg-white"
                : "cursor-not-allowed bg-white/[0.07] text-loma-faint",
            )}
          >
            {recap ? (
              <>
                <Sparkles className="size-3.5" />
                {busy ? "saving…" : "start chatting"}
              </>
            ) : (
              <>
                {isLast ? "review" : "next"}
                <ArrowRight className="size-3.5" />
              </>
            )}
          </button>
        </div>

        <p className="mt-3 text-center font-mono text-[10px] text-loma-faint/70">
          written to MEMORY.md on this machine — nothing is sent anywhere
        </p>
      </div>
    </div>
  )
}
