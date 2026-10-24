"""First-run answers become memory facts.

The wizard is not a form for its own sake: every answer has to survive as a line
in MEMORY.md that the model can act on, and a skipped question must produce
nothing at all.
"""

from __future__ import annotations

import json

from loma.onboarding import (
    completed_state,
    facts_from,
    read_state,
    skipped_state,
    summarise,
    write_state,
)


class TestFacts:
    def test_every_answer_becomes_a_fact(self):
        facts = facts_from(
            {
                "name": "Hamish",
                "role": "runs a small hardware channel",
                "goals": ["writing", "coding"],
                "style": "short and direct",
                "notes": "hard of hearing, prefers reading",
            }
        )
        assert facts == [
            "Prefers to be called Hamish",
            "Work and situation: runs a small hardware channel",
            "Wants help with writing, coding",
            "Likes answers that are short and direct",
            "Context to keep in mind: hard of hearing, prefers reading",
        ]

    def test_empty_answers_produce_nothing(self):
        assert facts_from({}) == []
        assert facts_from({"name": "", "role": "   ", "goals": [], "style": ""}) == []

    def test_a_long_answer_is_cut_at_a_word(self):
        fact = facts_from({"role": "x " * 400})[0]
        assert len(fact) < 300
        assert not fact.endswith(" ")
        assert "Work and situation: x" in fact

    def test_free_text_goals_are_allowed(self):
        assert facts_from({"goals": "debugging python"}) == [
            "Wants help with debugging python"
        ]

    def test_apostrophes_and_punctuation_survive(self):
        facts = facts_from({"name": "O'Brien."})
        assert facts == ["Prefers to be called O'Brien"]


class TestState:
    def test_completed_state_records_the_name(self, tmp_path):
        write_state(tmp_path, completed_state({"name": "Sam"}, 3))
        state = read_state(tmp_path)
        assert state["onboarded"] is True
        assert state["skipped"] is False
        assert state["name"] == "Sam"
        assert state["facts_stored"] == 3

    def test_skipping_is_remembered(self, tmp_path):
        write_state(tmp_path, skipped_state())
        state = read_state(tmp_path)
        assert state["onboarded"] is True
        assert state["skipped"] is True
        assert state["name"] == ""

    def test_a_missing_file_reads_as_empty(self, tmp_path):
        assert read_state(tmp_path) == {}

    def test_unreadable_json_reads_as_empty(self, tmp_path):
        (tmp_path / "onboarding.json").write_text("{not json")
        assert read_state(tmp_path) == {}


class TestSummary:
    def test_summary_is_what_the_recap_screen_shows(self):
        summary = summarise(
            {"name": " Sam ", "goals": ["a", " ", "b"], "style": "", "role": "dev"}
        )
        assert summary == {
            "name": "Sam",
            "role": "dev",
            "goals": ["a", "b"],
            "style": "",
            "notes": "",
        }


def test_state_round_trips_through_the_file(tmp_path):
    write_state(tmp_path, {"onboarded": True, "name": "Ada"})
    assert json.loads((tmp_path / "onboarding.json").read_text())["name"] == "Ada"
