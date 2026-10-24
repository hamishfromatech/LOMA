---
name: write-a-skill
description: Write a SKILL.md that LOMA loads on demand. Use when the user wants to teach LOMA a procedure or a house style.
---

# Write a skill

A skill is procedure, not knowledge. It tells LOMA *how to do a task* — the steps,
the order, the traps — and it is loaded only when the task comes up.

```
skills/
  my-skill/
    SKILL.md
```

## The file

```markdown
---
name: my-skill
description: One sentence. Says what it does and, critically, when to use it.
---

# My skill

## 1. First step
...
```

Two things are read at different times, and both have a job:

- **`description`** is in the context on **every turn**, as a one-line entry in the
  skill index. Write it as a trigger, not a summary: "Use when the user wants to
  split a file into chunks" routes; "A guide to chunking" does not.
- **The body** is loaded **only when the router asks for it**. Free to be detailed
  — but capped at `LOMA_SKILL_MAX_CHARS` (6000 by default), after which it is
  truncated with a pointer to the file.

## What makes a skill work on a 0.8B model

The model reading your skill is small. It follows structure well and nuance badly.

**Number the steps.** A numbered list is followed; a paragraph of prose is
skimmed and summarised back at the user.

**One instruction per step.** If a step has an "and", it is two steps.

**Say the order explicitly.** "First… then… never… " beats a set of facts the
model has to sequence itself.

**Give the exact strings.** Commands, file paths, field names. A small model will
not invent a correct path, but it will copy one perfectly.

**Name the failure.** "If the port is already in use, stop and say so" prevents
the model improvising a fix you did not authorise.

**No tables, no nested lists.** They survive the Markdown but not the small model.

## Length

Aim for 40–120 lines. Under that, the index line probably carries it already.
Over that, split it into two skills — the router can only load one per turn, so a
skill that covers three tasks will be loaded for the wrong one.

## Test it

```bash
loma skills                          # is it in the index?
loma skills --show my-skill          # is the body what you wrote?
loma ask "…something that should trigger it…" --no-speak
```

If it never loads, the description is the problem. If it loads but the answer is
wrong, the steps are. `loma route "…"` shows the router's choice without running
the language model at all.

## Do not

- Do not put facts about the user in a skill — that is `MEMORY.md`.
- Do not put tool definitions in a skill — those are MCP servers.
- Do not tell the user what you are doing ("I am now loading a skill"). Just do it.
