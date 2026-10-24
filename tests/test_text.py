"""The guards that keep a reasoning model's scratchpad out of the answer.

The failure this prevents is not cosmetic: LOMA speaks every sentence as it
arrives, so leaked reasoning is read aloud in the assistant's voice.
"""

from __future__ import annotations

import pytest

from loma.text import CLOSE, OPEN, ThinkingSplitter, strip_thinking


class TestStripThinking:
    def test_keeps_the_answer_after_the_marker(self):
        text = "The user is asking about their machine. I should be brief." + CLOSE + "You're on an M2."
        assert strip_thinking(text) == "You're on an M2."

    def test_keeps_a_clean_answer_untouched(self):
        text = "You're on an Apple M2 with eight gigabytes of memory."
        assert strip_thinking(text) == text

    def test_drops_a_block_with_no_closing_marker(self):
        assert strip_thinking("The user is asking something odd. Let me think.") == (
            "The user is asking something odd. Let me think."
        )

    def test_empty_opening_tag_with_nothing_inside_becomes_empty(self):
        assert strip_thinking(OPEN + "   ") == ""

    def test_last_marker_wins(self):
        text = f"a{CLOSE}b{CLOSE}c"
        assert strip_thinking(text) == "c"

    def test_leading_whitespace_is_dropped(self):
        assert strip_thinking(f"{CLOSE}\n\n  hello") == "hello"


class TestThinkingSplitter:
    def test_clean_answer_streams_after_the_lookahead(self):
        splitter = ThinkingSplitter(lookahead=10)
        out = "".join(splitter.push(chunk) for chunk in ["You ", "are ", "on ", "an M2 ", "today."])
        out += splitter.flush()
        assert out == "You are on an M2 today."

    def test_a_leak_inside_the_window_is_dropped(self):
        splitter = ThinkingSplitter(lookahead=1000)
        leaked = "The user wants specs. " + CLOSE
        emitted = "".join(splitter.push(part) for part in [leaked, "You're on an M2."])
        emitted += splitter.flush()
        assert emitted == "You're on an M2."
        assert "user wants" not in emitted
        assert splitter.saw_marker

    def test_marker_split_across_deltas_is_still_caught(self):
        splitter = ThinkingSplitter(lookahead=1000)
        emitted = "".join(splitter.push(part) for part in ["reasoning</th", "ink>the answer"])
        emitted += splitter.flush()
        assert emitted == "the answer"

    def test_an_opening_tag_holds_the_stream(self):
        splitter = ThinkingSplitter(lookahead=4)
        assert splitter.push(OPEN) == ""
        # Nothing is emitted while the block is open, however long it runs.
        for _ in range(50):
            assert splitter.push("more reasoning. ") == ""

    def test_committed_stream_passes_through(self):
        splitter = ThinkingSplitter(lookahead=4)
        splitter.push("Fine.")
        assert splitter.push(" Next sentence.") == " Next sentence."

    def test_flush_returns_the_tail_of_a_short_answer(self):
        splitter = ThinkingSplitter(lookahead=10_000)
        # No sentence end and inside the lookahead: it waits rather than guessing.
        assert splitter.push("Short") == ""
        assert splitter.flush() == "Short"

    def test_a_complete_sentence_commits_early(self):
        splitter = ThinkingSplitter(lookahead=10_000)
        assert splitter.push("Short.") == "Short."

    @pytest.mark.parametrize("size", [1, 3, 7])
    def test_never_loses_characters_when_reassembled(self, size):
        answer = "The quick brown fox jumps over the lazy dog, twice."
        splitter = ThinkingSplitter(lookahead=size)
        out = "".join(splitter.push(answer[i : i + size]) for i in range(0, len(answer), size))
        out += splitter.flush()
        assert out == answer
