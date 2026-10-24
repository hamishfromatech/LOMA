from loma.chunking import SentenceChunker


def test_splits_on_sentence_end():
    chunker = SentenceChunker(min_chars=1)
    assert chunker.push("Hello there. How ") == ["Hello there."]
    assert chunker.push("are you?") == ["How are you?"]


def test_no_split_without_following_whitespace():
    chunker = SentenceChunker(min_chars=1)
    # "3.14" must survive: the dot is followed by a digit, not whitespace.
    assert chunker.push("Pi is 3.14") == []
    assert chunker.push("159 and rising") == []


def test_sentence_at_end_of_stream_is_complete():
    chunker = SentenceChunker(min_chars=1)
    assert chunker.push("Hello") == []
    assert chunker.push(".") == ["Hello."]


def test_does_not_split_decimals():
    chunker = SentenceChunker(min_chars=1)
    out = chunker.push("Pi is 3.14159 and that is fine.")
    assert out == ["Pi is 3.14159 and that is fine."]


def test_does_not_split_abbreviations():
    chunker = SentenceChunker(min_chars=1)
    out = chunker.push("Dr. Smith arrived. He was late.")
    assert out == ["Dr. Smith arrived.", "He was late."]


def test_flush_returns_remainder():
    chunker = SentenceChunker(min_chars=1)
    chunker.push("no punctuation here")
    assert chunker.flush() == ["no punctuation here"]
    assert chunker.flush() == []


def test_min_chars_merges_short_sentences():
    chunker = SentenceChunker(min_chars=20)
    # "Hi." is too short to be spoken on its own, so it rides along with the
    # next sentence instead of costing a separate TTS round trip.
    assert chunker.push("Hi. ") == []
    out = chunker.push("This is a much longer sentence.")
    assert out == ["Hi. This is a much longer sentence."]


def test_max_chars_forces_a_split():
    chunker = SentenceChunker(min_chars=1, max_chars=20)
    out = chunker.push("word " * 12)
    assert out
    assert all(len(s) <= 20 for s in out)


def test_trailing_quote_is_kept_with_the_sentence():
    chunker = SentenceChunker(min_chars=1)
    assert chunker.push('He said "done." Then he left.') == ['He said "done."', "Then he left."]
