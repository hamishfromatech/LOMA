from loma.config import Settings
from loma.decision import DecisionEngine
from loma.ollama_client import OllamaError


def make_response(choice="answer", confidence=0.9, probs=None, reasoning=None):
    probabilities = probs or {choice: 0.9, "answer": 0.05, "stop": 0.02}
    answers = {
        "intent": {
            "type": "choice",
            "choice": choice,
            "probabilities": probabilities,
            "confidence": confidence,
        }
    }
    if reasoning is not None:
        answers["needs_reasoning"] = {"type": "noul", "noul": reasoning}
    return {"model": "tev1:0.8b-q8_0", "answers": answers, "usage": {}}


class FakeClient:
    def __init__(self, response=None, error=None):
        self.response = response or make_response()
        self.error = error
        self.calls = []

    def systemone(self, state, questions, *, keep_alive=None):
        self.calls.append((state, questions))
        if self.error:
            raise self.error
        return self.response


def test_high_confidence_control_intent_is_a_fast_action():
    client = FakeClient(make_response("stop", 0.95, {"stop": 0.97, "answer": 0.02}))
    decision = DecisionEngine(client, Settings()).decide("stop talking")
    assert decision.intent == "stop"
    assert decision.is_fast_action
    assert decision.probability == 0.97
    assert decision.fallback_reason == ""


def test_adjacent_fast_actions_can_win_on_probability_alone():
    # "stop" vs "exit": confidence is low because the runner-up is close, but
    # both outcomes are cheap non-generative actions, so the winner stands.
    client = FakeClient(make_response("stop", 0.48, {"stop": 0.59, "exit": 0.38}))
    decision = DecisionEngine(client, Settings()).decide("stop")
    assert decision.intent == "stop"
    assert decision.is_fast_action
    assert decision.fallback_reason == ""


def test_low_confidence_with_an_answer_runner_up_abstains():
    client = FakeClient(make_response("stop", 0.30, {"stop": 0.30, "answer": 0.60}))
    decision = DecisionEngine(client, Settings()).decide("what time is it")
    assert decision.intent == "answer"
    assert "low confidence" in decision.fallback_reason


def test_low_confidence_abstains_to_answer():
    client = FakeClient(make_response("stop", 0.30, {"stop": 0.30, "answer": 0.60}))
    decision = DecisionEngine(client, Settings()).decide("what time is it")
    assert decision.intent == "answer"
    assert "low confidence" in decision.fallback_reason


def test_answer_intent_is_never_treated_as_an_action():
    client = FakeClient(make_response("answer", 0.99, {"answer": 0.99}))
    decision = DecisionEngine(client, Settings()).decide("tell me a joke")
    assert decision.intent == "answer"
    assert not decision.is_fast_action


def test_router_error_falls_back_instead_of_raising():
    client = FakeClient(error=OllamaError("model not found"))
    decision = DecisionEngine(client, Settings()).decide("hello")
    assert decision.intent == "answer"
    assert "router error" in decision.fallback_reason


def test_disabled_router_never_calls_the_model():
    client = FakeClient()
    engine = DecisionEngine(client, Settings(decision_enabled=False))
    decision = engine.decide("stop")
    assert decision.intent == "answer"
    assert client.calls == []


def test_empty_text_skips_the_router():
    client = FakeClient()
    decision = DecisionEngine(client, Settings()).decide("   ")
    assert decision.intent == "answer"
    assert client.calls == []


def test_adaptive_thinking_adds_a_second_question():
    client = FakeClient(make_response("answer", 0.9, reasoning=0.92))
    engine = DecisionEngine(client, Settings(adaptive_thinking=True))
    decision = engine.decide("why is the sky blue?")
    assert decision.needs_reasoning is True
    _, questions = client.calls[0]
    assert set(questions) == {"intent", "needs_reasoning"}
    assert questions["needs_reasoning"]["type"] == "noul"


def test_no_reasoning_question_by_default():
    client = FakeClient()
    engine = DecisionEngine(client, Settings())
    engine.decide("hello")
    _, questions = client.calls[0]
    assert set(questions) == {"intent"}


def test_choice_question_offers_every_intent():
    client = FakeClient()
    DecisionEngine(client, Settings()).decide("hi")
    _, questions = client.calls[0]
    assert questions["intent"]["type"] == "choice"
    assert "stop" in questions["intent"]["criteria"]
    assert "reset" in questions["intent"]["criteria"]
    assert "repeat" in questions["intent"]["criteria"]
    assert "exit" in questions["intent"]["criteria"]


def test_as_dict_is_json_shaped():
    client = FakeClient(make_response("reset", 0.9, {"reset": 0.9, "answer": 0.05}))
    decision = DecisionEngine(client, Settings()).decide("start over")
    data = decision.as_dict()
    assert data["intent"] == "reset"
    assert "probabilities" in data
    assert isinstance(data["confidence"], float)
