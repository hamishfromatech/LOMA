---
name: write-an-mcp-server
description: Build a FastMCP server LOMA can call. Use when the user wants a new tool, a new capability, or asks how to extend LOMA.
---

# Write an MCP server for LOMA

A tool is a Python function with a docstring. LOMA reads the docstring, shows it
to Needle (a 35 MB router), and Needle decides when to call it and fills the
arguments. Nothing about the tool reaches the language model, so the docstring is
the *entire* interface the router reasons about.

## 1. The file

Put it anywhere; `examples/mcp_local_tools.py` is the reference.

```python
from fastmcp import FastMCP

mcp = FastMCP("home")


@mcp.tool
def room_temperature(room: str) -> str:
    """Read the current temperature of a room.

    Args:
        room: The room name, such as "kitchen" or "study".
    """
    return f"{room}: 21.4 C"


if __name__ == "__main__":
    mcp.run()
```

## 2. The four rules that decide whether it works

**Always include an `Args:` section.** FastMCP only emits argument descriptions
when the docstring has one, and Needle reads those descriptions to fill values.
A tool without `Args:` is a tool that guesses.

**One tool per action.** `roll_die(sides)` is right every time;
`roll_dice(sides, count)` swapped its two numbers because both are integers and
neither name appears in the sentence. If two arguments have the same type,
split the tool or make one of them an enum.

**One sentence for `description`.** It becomes the tool's description verbatim.
Say what it does and when to use it — not how it is implemented.

**Constrain anything finite with an enum.** `Literal["light", "dark"]` or
`{"type": "string", "enum": [...]}` cannot be mis-filled; a free-text string can.

## 3. Register it

`mcp.json`, next to the shipped demo server:

```json
{
  "mcpServers": {
    "home": {
      "command": "${LOMA_PYTHON}",
      "args": ["${LOMA_ROOT}/tools/home.py"]
    }
  }
}
```

`${LOMA_ROOT}` and `${LOMA_PYTHON}` are expanded before the server starts, so the
path works on any machine.

## 4. Prove it routes

```bash
loma tools                          # is the tool listed, with its schema?
loma tools --probe "how warm is the kitchen"
loma ask "how warm is the kitchen" --no-speak
```

`loma tools --probe` runs the router without the language model. If the probe
picks the wrong tool, the docstring is wrong — not the model. Fix the wording and
probe again; it costs nothing and needs no download.

## 5. Failure modes and what they mean

| Symptom | Cause |
|---|---|
| Tool never chosen | Description too vague, or the verb is buried in the second sentence |
| Chosen but arguments empty | No `Args:` section in the docstring |
| Two numbers swapped | Two same-typed arguments — split the tool |
| Chosen for everything | Description is too broad ("does anything with text") |
| Works in `tools` but not `ask` | Confidence below `LOMA_TOOL_MIN_CONFIDENCE` (0.50) |

Keep the whole server offline. A tool that fetches is a broken promise, and this
assistant's whole point is that it cannot.
