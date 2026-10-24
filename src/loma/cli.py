"""The ``loma`` command line.

Subcommands are chosen so a student can explore one stage at a time:

    loma doctor              is everything installed and reachable?
    loma devices             which microphone and speakers exist?
    loma context             context window policy and current usage
    loma tools               list MCP tools and test the tool router
    loma route "hello"       control lexicon + Needle tool routing
    loma transcribe x.wav    speech -> text only
    loma speak "hello"       text -> speech only
    loma ask "hello"         text -> route -> tools -> LLM -> speech
    loma listen              the whole loop
    loma bench               prove the RAM and latency numbers
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from loma import __version__
from loma.config import Settings
from loma.context import describe_policy
from loma.control import ControlRouter
from loma.memory import format_snapshot, peak_rss_mb, snapshot, wait_for_settle
from loma.ollama_client import OllamaClient, OllamaError

log = logging.getLogger("loma")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    settings = Settings.from_env()
    handler = {
        "listen": cmd_listen,
        "ask": cmd_ask,
        "route": cmd_route,
        "tools": cmd_tools,
        "skills": cmd_skills,
        "context": cmd_context,
        "transcribe": cmd_transcribe,
        "speak": cmd_speak,
        "doctor": cmd_doctor,
        "devices": cmd_devices,
        "bench": cmd_bench,
        "models": cmd_models,
    }[args.command]

    try:
        return handler(args, settings)
    except OllamaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loma",
        description="LOMA - Local Offline Mini Assistant.",
    )
    parser.add_argument("--version", action="version", version=f"loma {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    # listen ----------------------------------------------------------
    listen = sub.add_parser("listen", help="run the full voice loop")
    listen.add_argument("--turns", type=int, default=None, help="stop after N turns")
    listen.add_argument("--no-warmup", action="store_true", help="skip model preloading")

    # ask -------------------------------------------------------------
    ask = sub.add_parser("ask", help="text prompt -> route -> tools -> LLM -> speech")
    ask.add_argument("prompt", nargs="+", help="what to ask")
    ask.add_argument("--no-speak", action="store_true", help="do not play audio")
    ask.add_argument("-o", "--out", type=Path, help="also save the spoken reply as WAV")
    ask.add_argument("--no-route", action="store_true", help="bypass the control lexicon")
    ask.add_argument("--no-tools", action="store_true", help="disable MCP tools")
    ask.add_argument("--no-warmup", action="store_true")

    # route -----------------------------------------------------------
    route = sub.add_parser(
        "route", aliases=["decide"], help="control lexicon + Needle tool routing, no LLM"
    )
    route.add_argument("text", nargs="+")
    route.add_argument("--no-tools", action="store_true", help="skip MCP tool routing")
    route.add_argument("--json", action="store_true")

    # tools -----------------------------------------------------------
    tools = sub.add_parser("tools", help="list MCP tools and test the router")
    tools.add_argument("--probe", metavar="TEXT", default=None, help="route this sentence")
    tools.add_argument("--json", action="store_true")

    # skills ----------------------------------------------------------
    skills = sub.add_parser("skills", help="list installed skills, or show one")
    skills.add_argument("--show", metavar="NAME", default=None, help="print a skill's body")
    skills.add_argument("--json", action="store_true")

    # context ---------------------------------------------------------
    context = sub.add_parser("context", help="show the context window policy")
    context.add_argument("--json", action="store_true")

    # transcribe ------------------------------------------------------
    transcribe = sub.add_parser("transcribe", help="transcribe an audio file")
    transcribe.add_argument("file", type=Path)

    # speak -----------------------------------------------------------
    speak = sub.add_parser("speak", help="text to speech only")
    speak.add_argument("text", nargs="+")
    speak.add_argument("--voice", default=None, help="Phonon TTS voice name, e.g. Zoey")
    speak.add_argument("-o", "--out", type=Path, help="save WAV instead of playing")
    speak.add_argument("--raw", action="store_true", help="write raw PCM instead of WAV")

    # doctor ----------------------------------------------------------
    doctor = sub.add_parser("doctor", help="check the installation end to end")
    doctor.add_argument("--pull", action="store_true", help="pull missing Ollama models")
    doctor.add_argument("--json", action="store_true", help="machine-readable output")

    # devices ---------------------------------------------------------
    sub.add_parser("devices", help="list audio devices")

    # bench -----------------------------------------------------------
    bench = sub.add_parser("bench", help="measure latency and RAM")
    bench.add_argument("--turns", type=int, default=3)
    bench.add_argument("--json", action="store_true")
    bench.add_argument("--allow-over-budget", action="store_true")

    # models ----------------------------------------------------------
    sub.add_parser("models", help="show which models are loaded in memory")
    return parser


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# listen / ask
# ---------------------------------------------------------------------------


def cmd_listen(args, settings: Settings) -> int:
    from loma.pipeline import VoicePipeline

    with VoicePipeline(settings) as pipeline:
        if not args.no_warmup:
            pipeline.warm_up()
        pipeline.listen_loop(max_turns=args.turns)
    return 0


def cmd_ask(args, settings: Settings) -> int:
    from loma.pipeline import StopLoop, VoicePipeline

    prompt = " ".join(args.prompt)
    with VoicePipeline(settings) as pipeline:
        if not args.no_warmup:
            pipeline.warm_up(speech=not args.no_speak)
        try:
            result = pipeline.respond(
                prompt,
                speak=not args.no_speak,
                save_wav=args.out,
                route=not args.no_route,
                use_tools=not args.no_tools,
                on_delta=(lambda d: sys.stdout.write(d)) if not args.no_speak else None,
                on_sentence=(lambda s: print(f"\n  [speaking] {s}")) if not args.no_speak else None,
            )
        except StopLoop:
            return 0

    if not args.no_speak and result.reply:
        print()
    print(result.summary())
    return 0


# ---------------------------------------------------------------------------
# route / tools / context
# ---------------------------------------------------------------------------


def _route_payload(text: str, settings: Settings, *, with_tools: bool) -> dict:
    """Control match plus (optionally) a live Needle routing decision."""
    from loma.mcp_hub import MCPHub, load_mcp_config
    from loma.tool_router import NeedleRouter

    control = ControlRouter(settings.control_enabled).match(text)
    payload: dict = {"control": control.as_dict()}

    if not with_tools or not settings.mcp_enabled:
        return payload

    hub = MCPHub(load_mcp_config(settings.mcp_config), max_tools=settings.tool_max_tools)
    router = NeedleRouter(settings)
    try:
        if hub.start() and router.load(hub.tools):
            payload["tools"] = router.route(text).as_dict()
        else:
            payload["tools"] = {"error": hub.last_error or router.last_error}
    finally:
        router.close()
        hub.stop()
    return payload


def cmd_route(args, settings: Settings) -> int:
    text = " ".join(args.text)
    payload = _route_payload(text, settings, with_tools=not args.no_tools)

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    control = payload["control"]
    print(f"text    : {text}")
    print(f"control : {control['intent']}  ({control['latency_s'] * 1000:.2f} ms)")
    if control["matched"]:
        print(f"matched : {control['matched']!r}")

    tool_info = payload.get("tools")
    if tool_info:
        print()
        if "error" in tool_info:
            print(f"tools   : unavailable ({tool_info['error']})")
        elif tool_info["tools"]:
            calls = ", ".join(
                f"{c['tool']}({c['arguments']})" for c in tool_info["tools"]
            )
            print(f"tools   : {calls}")
            print(f"needle  : confidence {tool_info['confidence']:.2f} "
                  f"in {tool_info['latency_s'] * 1000:.0f} ms")
            if tool_info.get("reasoning"):
                print(f"why     : {tool_info['reasoning']}")
        else:
            print(f"tools   : none  ({tool_info['reason']})")
            print(f"needle  : confidence {tool_info['confidence']:.2f} "
                  f"in {tool_info['latency_s'] * 1000:.0f} ms")
    return 0


def cmd_tools(args, settings: Settings) -> int:
    from loma.mcp_hub import MCPHub, load_mcp_config
    from loma.tool_router import NeedleRouter

    if not settings.mcp_config.exists():
        print(f"no MCP config at {settings.mcp_config}")
        print("Create mcp.json (see mcp.json.example) to enable tools.")
        return 1

    hub = MCPHub(load_mcp_config(settings.mcp_config), max_tools=settings.tool_max_tools)
    router = NeedleRouter(settings)
    try:
        if not hub.start():
            print(f"MCP could not start: {hub.last_error or 'unknown error'}", file=sys.stderr)
            return 1

        if args.probe:
            if not router.load(hub.tools):
                print(f"Needle could not start: {router.last_error}", file=sys.stderr)
                return 1
            route = router.route(args.probe)
            if args.json:
                print(json.dumps(route.as_dict(), indent=2))
                return 0
            print(f"probe   : {args.probe}")
            if route.requests:
                for request in route.requests:
                    print(f"tool    : {request.tool.name}")
                    print(f"args    : {request.arguments}")
            else:
                print(f"tool    : none  ({route.reason})")
            print(f"needle  : confidence {route.confidence:.2f} "
                  f"in {route.latency_s * 1000:.0f} ms")
            if route.reasoning:
                print(f"why     : {route.reasoning}")
            if route.suppressed:
                print(f"held    : {route.suppressed} call(s) below the confidence floor")
            return 0

        if args.json:
            print(json.dumps(
                [
                    {
                        "name": t.name,
                        "description": t.description,
                        "required": t.required,
                        "server": t.server,
                    }
                    for t in hub.tools
                ],
                indent=2,
            ))
            return 0

        print(f"{len(hub.tools)} tool(s) from {settings.mcp_config}:")
        for tool in hub.tools:
            required = ", ".join(tool.required) or "-"
            print(f"\n  {tool.name}")
            print(f"    {tool.short_description}")
            print(f"    required: {required}")
        print('\nRoute a sentence with: loma tools --probe "what time is it"')
        return 0
    finally:
        router.close()
        hub.stop()


def cmd_skills(args, settings: Settings) -> int:
    """List the skills LOMA can load, or print one in full."""
    from loma.skills import SkillStore

    if not settings.skills_enabled:
        print("skills are disabled (LOMA_SKILLS_ENABLED=0)")
        return 1

    store = SkillStore(settings.skills_root, max_chars=settings.skill_max_chars)

    if args.show:
        body = store.load(args.show)
        if not body:
            available = ", ".join(store.names()) or "none"
            print(f"no skill named {args.show!r}. Installed: {available}", file=sys.stderr)
            return 1
        print(body)
        return 0

    skills = store.scan()
    if not skills:
        print(f"no skills installed in {settings.skills_root}")
        print("Add skills/<name>/SKILL.md — see skills/write-a-skill/SKILL.md.")
        return 1

    if args.json:
        print(json.dumps(
            [
                {"name": s.name, "description": s.description, "path": str(s.path)}
                for s in skills
            ],
            indent=2,
        ))
        return 0

    print(f"{len(skills)} skill(s) in {settings.skills_root}")
    width = max(len(s.name) for s in skills)
    for skill in skills:
        print(f"  {skill.name:<{width}}  {skill.short_description}")
    print()
    print("The index above is in every prompt; a body is loaded only when asked for.")
    print("Show one with: loma skills --show <name>")
    return 0


def cmd_context(args, settings: Settings) -> int:
    info = {
        "num_ctx": settings.num_ctx,
        "policy": describe_policy(),
        "compact_enabled": settings.compact_enabled,
        "compact_at": settings.compact_at,
        "compact_target_tokens": settings.compact_target_tokens,
        "compact_threshold_tokens": settings.compact_threshold_tokens,
        "compact_keep_turns": settings.compact_keep_turns,
        "history_turns": settings.history_turns,
    }
    if args.json:
        print(json.dumps(info, indent=2))
        return 0

    print(describe_policy())
    print(f"context window   : {settings.num_ctx:,} tokens")
    print(f"compaction at    : {min(settings.compact_at, settings.compact_target_tokens / settings.num_ctx):.0%} "
          f"({settings.compact_threshold_tokens:,} live tokens)")
    print(f"keep verbatim    : last {settings.compact_keep_turns} exchange(s)")
    print(f"hard history cap : {settings.history_turns} exchanges")
    print()
    print("The 128k window is a ceiling, not a target: compaction keeps live")
    print("context near the target, which is what keeps the KV cache small.")
    print("Verify the real footprint with: loma bench")
    return 0


# ---------------------------------------------------------------------------
# transcribe / speak
# ---------------------------------------------------------------------------


def cmd_transcribe(args, settings: Settings) -> int:
    from loma.stt import Transcriber

    if not args.file.exists():
        print(f"error: no such file: {args.file}", file=sys.stderr)
        return 2
    transcriber = Transcriber(settings)
    result = transcriber.transcribe(args.file)
    print(result.text)
    log.debug("language=%s duration=%.2fs", result.language, result.duration)
    return 0


def cmd_speak(args, settings: Settings) -> int:
    from loma.audio import play_pcm, write_wav

    text = " ".join(args.text)
    with OllamaClient(settings) as client:
        if args.out:
            if args.raw:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                chunks = list(client.speech_pcm(text, voice=args.voice))
                args.out.write_bytes(b"".join(chunks))
            else:
                write_wav(args.out, client.speech_wav(text, voice=args.voice))
            print(f"wrote {args.out}")
        else:
            chunks = list(client.speech_pcm(text, voice=args.voice))
            play_pcm(b"".join(chunks), sample_rate=settings.tts_sample_rate)
    return 0


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def _needle_cache_ready() -> bool:
    cache = Path.home() / ".cache" / "cactus-needle"
    return any(cache.rglob("needle3.cact"))


def cmd_doctor(args, settings: Settings) -> int:
    import shutil

    checks: list[tuple[str, bool, str]] = []

    # Ollama reachable + models
    version = "unknown"
    try:
        with OllamaClient(settings, timeout=10) as client:
            version = client.health().get("version", "unknown")
            checks.append(("Ollama reachable", True, f"v{version} at {settings.ollama_host}"))

            required = [settings.llm_model, settings.tts_model]
            missing = []
            for name in required:
                present = client.has_model(name)
                checks.append((f"model {name}", present, "installed" if present else "missing"))
                if not present:
                    missing.append(name)

            if missing and args.pull:
                import subprocess

                for name in missing:
                    log.info("pulling %s", name)
                    subprocess.run(["ollama", "pull", name], check=False)
    except OllamaError as exc:
        checks.append(("Ollama reachable", False, str(exc)))

    # faster-whisper
    from loma.stt import Transcriber

    stt_ok = Transcriber.available()
    checks.append(("faster-whisper installed", stt_ok, settings.stt_model))
    if stt_ok:
        cached = settings.whisper_dir / f"models--Systran--faster-whisper-{settings.stt_model}"
        checks.append(("whisper weights cached", cached.exists(), str(settings.whisper_dir)))

    # Needle tool router
    try:
        import needle  # noqa: F401

        checks.append(("Needle installed", True, "cactus-needle"))
        checks.append(("Needle weights cached", _needle_cache_ready(), "~/.cache/cactus-needle"))
    except ImportError:
        checks.append(("Needle installed", False, "pip install cactus-needle"))

    # Control lexicon (always available; this documents that no model is needed)
    checks.append(("control lexicon", settings.control_enabled, "deterministic, 0 MB"))

    # audio
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        inputs = sum(1 for d in devices if d["max_input_channels"] > 0)
        outputs = sum(1 for d in devices if d["max_output_channels"] > 0)
        checks.append(("audio devices", inputs > 0 and outputs > 0, f"{inputs} in / {outputs} out"))
    except Exception as exc:  # noqa: BLE001
        checks.append(("audio devices", False, f"sounddevice unavailable: {exc}"))

    # MCP
    from loma.mcp_hub import MCPHub, load_mcp_config

    config = load_mcp_config(settings.mcp_config) if settings.mcp_enabled else None
    if config is None:
        checks.append(("MCP servers", True, f"none configured ({settings.mcp_config})"))
    else:
        hub = MCPHub(config, max_tools=settings.tool_max_tools)
        try:
            ok = hub.start()
            detail = (
                f"{len(hub.tools)} tools from {len(config['mcpServers'])} server(s)"
                if ok
                else hub.last_error or "failed to start"
            )
            checks.append(("MCP servers", ok, detail))
        finally:
            hub.stop()

    # context policy
    checks.append(
        ("context window", settings.num_ctx >= 131_072, f"{settings.num_ctx // 1024}k tokens")
    )

    # optional tools
    checks.append(("ffmpeg (optional)", shutil.which("ffmpeg") is not None, "for odd input formats"))

    # memory
    snap = snapshot()
    checks.append(
        (
            "RAM headroom",
            snap.total_mb <= settings.memory_budget_mb,
            f"{snap.total_mb:.0f} MB of {settings.memory_budget_mb} MB used (idle)",
        )
    )

    if args.json:
        print(json.dumps(
            {
                "ok": all(ok for _, ok, _ in checks),
                "version": version,
                "checks": [{"name": n, "ok": o, "detail": d} for n, o, d in checks],
            },
            indent=2,
        ))
        return 0 if all(ok for _, ok, _ in checks) else 1

    width = max(len(name) for name, _, _ in checks)
    for name, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name:<{width}}  {detail}")

    failed = [name for name, ok, _ in checks if not ok]
    if failed:
        print(f"\n{len(failed)} check(s) failed: {', '.join(failed)}")
        print("See docs/TROUBLESHOOTING.md")
        return 1
    print("\nAll checks passed. Run `loma listen` to talk.")
    return 0


# ---------------------------------------------------------------------------
# devices / models
# ---------------------------------------------------------------------------


def cmd_devices(args, settings: Settings) -> int:
    from loma.audio import list_devices

    print(list_devices())
    print("\nPick one with LOMA_MIC_DEVICE=<index>")
    return 0


def cmd_models(args, settings: Settings) -> int:
    with OllamaClient(settings) as client:
        snap = snapshot(client)
    print(format_snapshot(snap, settings.memory_budget_mb))
    return 0


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------


def cmd_bench(args, settings: Settings) -> int:
    from loma.pipeline import VoicePipeline

    prompts = [
        "In one sentence, what is a voice agent?",
        "Give me a short fun fact about the ocean.",
        "What is two plus two? Answer briefly.",
    ]
    results = []
    control_times: list[float] = []
    tool_times: list[float] = []

    with VoicePipeline(settings) as pipeline:
        pipeline.warm_up()
        wait_for_settle(1.0)

        controls = ControlRouter(settings.control_enabled)
        for phrase in ("stop", "start over", "say that again", "goodbye"):
            control_times.append(controls.match(phrase).latency_s)

        if pipeline.start_tools() and pipeline.hub.tools:
            for phrase in ("what time is it", "roll two dice", "how are you"):
                tool_times.append(pipeline.tool_router.route(phrase).latency_s)

        for index in range(args.turns):
            prompt = prompts[index % len(prompts)]
            result = pipeline.reply(prompt, speak=False)
            results.append(result)
            print(f"turn {index + 1}: {result.total_s:.2f}s -> {result.reply[:80]}")

        wait_for_settle(1.0)
        snap = snapshot(pipeline.client)

    peak = peak_rss_mb()
    avg_total = sum(r.total_s for r in results) / len(results) if results else 0.0
    avg_first = sum(r.first_token_s for r in results) / len(results) if results else 0.0
    avg_control = sum(control_times) / len(control_times) if control_times else 0.0
    avg_tool = sum(tool_times) / len(tool_times) if tool_times else 0.0
    ok = snap.total_mb <= settings.memory_budget_mb

    if args.json:
        print(json.dumps(
            {
                "memory": snap.as_dict(),
                "python_peak_rss_mb": round(peak, 1),
                "num_ctx": settings.num_ctx,
                "avg_control_us": round(avg_control * 1_000_000, 1),
                "avg_tool_route_ms": round(avg_tool * 1000, 1),
                "avg_total_s": round(avg_total, 3),
                "avg_first_token_s": round(avg_first, 3),
                "budget_mb": settings.memory_budget_mb,
                "within_budget": ok,
            },
            indent=2,
        ))
    else:
        print()
        print(format_snapshot(snap, settings.memory_budget_mb))
        print(f"\nContext window  : {settings.num_ctx:,} tokens")
        print(f"Python peak RSS : {peak:.0f} MB")
        print(f"Avg control     : {avg_control * 1_000_000:.0f} us")
        if avg_tool:
            print(f"Avg tool route  : {avg_tool * 1000:.0f} ms  (Needle)")
        print(f"Avg first token : {avg_first:.2f}s")
        print(f"Avg full turn   : {avg_total:.2f}s")

    if not ok and not args.allow_over_budget:
        print("\nFAILED: over the memory budget", file=sys.stderr)
        print("See docs/RAM_BUDGET.md for the levers.", file=sys.stderr)
        return 1
    return 0
