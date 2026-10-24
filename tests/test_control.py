from loma.control import PHRASES, ControlRouter, normalise


def intent_of(text: str) -> str:
    return ControlRouter().match(text).intent


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_normalise_lowercases_and_strips_punctuation():
    assert normalise("Stop!") == "stop"
    assert normalise("  Goodbye.  ") == "goodbye"


def test_normalise_removes_filler_words():
    assert normalise("can you please just stop") == "stop"
    assert normalise("okay, be quiet") == "quiet"
    assert normalise("thanks, that's all") == "that's all"


def test_normalise_keeps_meaningful_words():
    assert normalise("what does reset mean") == "what does reset mean"


# ---------------------------------------------------------------------------
# Control phrases
# ---------------------------------------------------------------------------


def test_stop_phrasings():
    for text in [
        "stop", "Stop it, please.", "stop talking", "be quiet", "shut up",
        "never mind", "that's enough thanks", "hush", "cancel that",
    ]:
        assert intent_of(text) == "stop", text


def test_reset_phrasings():
    for text in [
        "start over", "start again", "forget everything", "reset",
        "clear the conversation", "new conversation", "fresh start",
    ]:
        assert intent_of(text) == "reset", text


def test_repeat_phrasings():
    for text in [
        "say that again", "repeat", "repeat that", "what did you say",
        "once more", "come again",
    ]:
        assert intent_of(text) == "repeat", text


def test_exit_phrasings():
    for text in ["goodbye", "quit", "exit", "bye", "thanks, that's all", "i'm done"]:
        assert intent_of(text) == "exit", text


def test_exit_wins_over_stop_when_both_appear():
    # "that's all" ends the session; "that's enough" only stops the turn.
    assert intent_of("that's all") == "exit"
    assert intent_of("that's enough") == "stop"


# ---------------------------------------------------------------------------
# Non-control utterances must fall through
# ---------------------------------------------------------------------------


def test_questions_are_not_control_phrases():
    for text in [
        "what is the capital of France?",
        "how are you?",
        "tell me a joke",
        "what does reset mean?",
        "don't stop believing",
        "please continue",
        "what time is it",
    ]:
        assert intent_of(text) == "answer", text


def test_empty_text_is_not_a_control_phrase():
    assert ControlRouter().match("").intent == "answer"
    assert ControlRouter().match("   ").intent == "answer"


def test_disabled_router_never_matches():
    router = ControlRouter(enabled=False)
    assert router.match("stop").intent == "answer"
    assert router.match("goodbye").intent == "answer"


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_match_reports_the_phrase_and_is_fast():
    match = ControlRouter().match("Can you please stop?")
    assert match.intent == "stop"
    assert match.matched == "stop"
    assert match.is_fast_action
    assert match.latency_s < 0.01


def test_answer_is_not_a_fast_action():
    assert not ControlRouter().match("hello").is_fast_action


def test_as_dict_is_json_shaped():
    data = ControlRouter().match("goodbye").as_dict()
    assert data["intent"] == "exit"
    assert "latency_s" in data


def test_every_intent_has_phrases():
    assert set(PHRASES) == {"stop", "reset", "repeat", "exit"}
    assert all(PHRASES[i] for i in PHRASES)
