"""A remembered fact must be supported by the user's own words.

The failure this prevents was observed, not imagined: asking for "a long story
about a lighthouse" got LOMA to store "The user lives in a coastal town where
they work as a local guide and enjoy hiking to the nearest lighthouse".
"""

from __future__ import annotations

from loma.memory_evidence import quote_appears


class TestQuoteAppears:
    def test_an_exact_quote_passes(self):
        assert quote_appears("I prefer tea", "I prefer tea over coffee")

    def test_case_and_whitespace_do_not_matter(self):
        assert quote_appears("i   PREFER\n tea", "I prefer tea over coffee")

    def test_punctuation_around_the_quote_is_tolerated(self):
        assert quote_appears("I prefer tea", "Well, I prefer tea, actually.")

    def test_an_invented_quote_fails(self):
        assert not quote_appears(
            "I live in a coastal town",
            "tell me a long story about a lighthouse",
        )

    def test_an_empty_quote_fails(self):
        assert not quote_appears("", "I prefer tea")

    def test_a_paraphrase_fails(self):
        assert not quote_appears("the user prefers tea", "I prefer tea over coffee")


class TestLooksLikeAFact:
    """A remembered fact is a statement about the person, not a sentence."""

    def test_a_plain_statement_passes(self):
        from loma.memory_evidence import looks_like_fact

        assert looks_like_fact("Your name is Hamish", "my name is Hamish")

    def test_a_question_is_not_a_fact(self):
        from loma.memory_evidence import looks_like_fact

        assert not looks_like_fact("What is your name?", "what is my name")
        assert not looks_like_fact("Is my name Hamish", "is my name Hamish")

    def test_a_quoted_question_is_not_a_fact(self):
        from loma.memory_evidence import looks_like_fact

        assert not looks_like_fact("Your name", "what is my name?")

    def test_meta_sentences_are_not_facts(self):
        from loma.memory_evidence import looks_like_fact

        assert not looks_like_fact("You asked what I know about you", "what do you know about me")

    def test_an_imperative_is_not_a_fact(self):
        from loma.memory_evidence import looks_like_fact

        assert not looks_like_fact("Tell me a story about a lighthouse", "tell me a story")
