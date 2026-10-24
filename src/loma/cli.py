"""The ``loma`` command line.

Subcommands are chosen so a student can explore one stage at a time:

    loma doctor              is everything installed and reachable?
    loma devices             which microphone and speakers exist?
    loma context             context window policy and current usage
    loma tools               list MCP tools and test the tool router
    loma decide "hello"      decision model only (~150 ms)
    loma transcribe x.wav    speech -> text only
    loma speak "hello"       text -> speech only
    loma ask "hello"         text -> router -> tools -> LLM -> speech
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
        "decide": cmd_decide,
        "tools": cmd_tools,
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
    ask = sub.add_parser("ask", help="text prompt -> router -> tools -> LLM -> speech")
    ask.add_argument("prompt", nargs="+", help="what to ask")
    ask.add_argument("--no-speak", action="store_true", help="do not play audio")
    ask.add_argument("-o", "--out", type=Path, help="also save the spoken reply as WAV")
    ask.add_argument("--no-route", action="store_true", help="bypass the decision router")
    ask.add_argument("--no-tools", action="store_true", help="disable MCP tools")
    ask.add_argument("--no-warmup", action="store_true")

    # decide ----------------------------------------------------------
    decide = sub.add_parser("decide", help="run the decision model on some text")
    decide.add_argument("text", nargs="+")
    decide.add_argument("--tools", action="store_true", help="also route to MCP tools")
    decide.add_argument("--json", action="store_true")

    # tools -----------------------------------------------------------
    tools = sub.add_parser("tools", help="list MCP tools and test the router")
    tools.add_argument("--probe", metavar="TEXT", default=None, help="route this sentence")
    tools.add_argument("--json", action="store_true")

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

    if args.no_speak or not result.reply:
        print(result.summary())
    else:
        print()
        print(result.summary())
    return 0


# ---------------------------------------------------------------------------
# decide / tools / context
# ---------------------------------------------------------------------------


def cmd_decide(args, settings: Settings) -> int:
    from loma.decision import DecisionEngine
    from loma.mcp_hub import MCPHub, load_mcp_config
    from loma.tool_router import ToolRouter

    text = " ".join(args.text)
    payload: dict = {}

    with OllamaClient(settings) as client:
        decision = DecisionEngine(client, settings).decide(text)
        payload["intent"] = decision.as_dict()

        if args.tools:
            hub = MCPHub(
                load_mcp_config(settings.mcp_config) if settings.mcp_enabled else None,
                max_tools=settings.tool_max_tools,
            )
            try:
                if hub.start():
                    tool_decision = ToolRouter(client, settings).choose(text, hub.tools)
                    payload["tool"] = tool_decision.as_dict()
            finally:
                hub.stop()

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"text        : {text}")
    print(f"intent      : {decision.intent}")
    print(f"probability : {decision.probability:.1%}")
    print(f"confidence  : {decision.confidence:.1%}")
    print(f"latency     : {decision.latency_s * 1000:.0f} ms")
    if decision.fallback_reason:
        print(f"fallback    : {decision.fallback_reason}")
    print("probabilities:")
    for name, probability in sorted(decision.probabilities.items(), key=lambda i: -i[1]):
        print(f"  {name:<10} {probability:6.1%}  {'#' * int(probability * 40)}")

    tool_info = payload.get("tool")
    if tool_info:
        print("\ntool route  :")
        print(f"  tool      : {tool_info['tool'] or 'none'}")
        print(f"  confidence: {tool_info['confidence']:.1%}")
        print(f"  latency   : {tool_info['latency_s'] * 1000:.0f} ms")
        if tool_info.get("reason"):
            print(f"  reason    : {tool_info['reason']}")
    return 0


def cmd_tools(args, settings: Settings) -> int:
    from loma.mcp_hub import MCPHub, load_mcp_config
    from loma.tool_router import ToolRouter

    if not settings.mcp_config.exists():
        print(f"no MCP config at {settings.mcp_config}")
        print("Create mcp.json (see mcp.json.example) to enable tools.")
        return 1

    hub = MCPHub(load_mcp_config(settings.mcp_config), max_tools=settings.tool_max_tools)
    try:
        if not hub.start():
            print(f"MCP could not start: {hub.last_error or 'unknown error'}", file=sys.stderr)
            return 1

        if args.probe:
            with OllamaClient(settings) as client:
                decision = ToolRouter(client, settings).choose(args.probe, hub.tools)
            if args.json:
                print(json.dumps({"tools": [t.name for t in hub.tools], "probe": decision.as_dict()}, indent=2))
            else:
                print(f"probe: {args.probe}")
                print(f"tool : {decision.tool.name if decision.tool else 'none'}")
                print(f"conf : {decision.confidence:.1%}  ({decision.latency_s * 1000:.0f} ms)")
                if decision.reason:
                    print(f"note : {decision.reason}")
                for name, probability in sorted(
                    decision.probabilities.items(), key=lambda i: -i[1]
                ):
                    print(f"  {name:<24} {probability:6.1%}")
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
        print("\nRoute a sentence with: loma tools --probe \"what time is it\"")
        return 0
    finally:
        hub.stop()


def cmd_context(args, settings: Settings) -> int:
    info = {
        "num_ctx": settings.num_ctx,
        "policy": describe_policy(),
        "compact_enabled": settings.compact_enabled,
        "compact_at": settings.compact_at,
        "compact_threshold_tokens": settings.compact_threshold_tokens,
        "compact_keep_turns": settings.compact_keep_turns,
        "history_turns": settings.history_turns,
    }
    if args.json:
        print(json.dumps(info, indent=2))
        return 0

    print(describe_policy())
    print(f"context window   : {settings.num_ctx:,} tokens")
    print(f"compaction at    : {settings.compact_at:.0%} "
          f"({settings.compact_threshold_tokens:,} tokens)")
    print(f"keep verbatim    : last {settings.compact_keep_turns} exchange(s)")
    print(f"hard history cap : {settings.history_turns} exchanges")
    print()
    print("A 128k window needs a quantised KV cache to stay affordable:")
    print("  export OLLAMA_KV_CACHE_TYPE=q4_0   # then restart Ollama")
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


def cmd_doctor(args, settings: Settings) -> int:
    import shutil

    checks: list[tuple[str, bool, str]] = []

    # Ollama reachable + models
    version = "unknown"
    loaded_ctx = None
    try:
        with OllamaClient(settings, timeout=10) as client:
            version = client.health().get("version", "unknown")
            checks.append(("Ollama reachable", True, f"v{version} at {settings.ollama_host}"))

            required = [settings.llm_model, settings.tts_model]
            if settings.decision_enabled:
                required.append(settings.decision_model)
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
        (
            "context window",
            settings.num_ctx >= 131_072,
            f"{settings.num_ctx // 1024}k tokens",
        )
    )
    checks.append(
        (
            "KV cache tip",
            True,
            "set OLLAMA_KV_CACHE_TYPE=q4_0 to make 128k fit (see docs/RAM_BUDGET.md)",
        )
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
                "context": loaded_ctx,
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
    decision_times: list[float] = []
    tool_times: list[float] = []

    with VoicePipeline(settings) as pipeline:
        pipeline.warm_up()
        wait_for_settle(1.0)

        for _ in range(3):
            decision = pipeline.decider.decide("what is the capital of France?")
            decision_times.append(decision.latency_s)

        if pipeline.hub.available and pipeline.hub.tools:
            for _ in range(3):
                choice = pipeline.tool_router.choose(
                    "what time is it right now", pipeline.hub.tools
                )
                tool_times.append(choice.latency_s)

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
    avg_decision = sum(decision_times) / len(decision_times) if decision_times else 0.0
    avg_tool = sum(tool_times) / len(tool_times) if tool_times else 0.0
    ok = snap.total_mb <= settings.memory_budget_mb

    if args.json:
        print(json.dumps(
            {
                "memory": snap.as_dict(),
                "python_peak_rss_mb": round(peak, 1),
                "num_ctx": settings.num_ctx,
                "avg_decision_ms": round(avg_decision * 1000, 1),
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
        print(f"Avg decision    : {avg_decision * 1000:.0f} ms")
        if avg_tool:
            print(f"Avg tool route  : {avg_tool * 1000:.0f} ms")
        print(f"Avg first token : {avg_first:.2f}s")
        print(f"Avg full turn   : {avg_total:.2f}s")

    if not ok and not args.allow_over_budget:
        print("\nFAILED: over the memory budget", file=sys.stderr)
        print("See docs/RAM_BUDGET.md for the levers.", file=sys.stderr)
        return 1
    return 0
