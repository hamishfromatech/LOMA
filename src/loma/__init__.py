"""LOMA — Local Offline Mini Assistant.

A voice agent that runs entirely on your own machine, under 3 GB of RAM.

    config.py        typed settings (env-driven)
    context.py       128k/256k window policy + token accounting
    ollama_client.py one HTTP client for the LLM, the TTS model and the router
    stt.py           faster-whisper speech-to-text
    audio.py         microphone capture, VAD, speaker playback
    decision.py      System One intent router (fast actions)
    tool_router.py   System One tool selection + argument extraction
    mcp_hub.py       FastMCP client bridge (sync API over async clients)
    agent.py         conversation state and auto-compaction
    pipeline.py      the end-to-end loop
    memory.py        RAM accounting against the budget
    cli.py           the `loma` command
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
