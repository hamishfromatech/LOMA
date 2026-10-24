from loma.context import (
    LARGE_CONTEXT_WINDOW,
    MIN_CONTEXT_WINDOW,
    context_window_for_ram,
    describe_policy,
    estimate_messages_tokens,
    estimate_tokens,
)


def test_eight_gigabyte_machine_gets_128k():
    assert context_window_for_ram(8.0) == MIN_CONTEXT_WINDOW


def test_smaller_machine_still_gets_128k():
    # The floor is a hard rule, not a scale.
    assert context_window_for_ram(4.0) == MIN_CONTEXT_WINDOW


def test_machine_above_eight_gigabytes_gets_256k():
    assert context_window_for_ram(8.1) == LARGE_CONTEXT_WINDOW
    assert context_window_for_ram(16.0) == LARGE_CONTEXT_WINDOW
    assert context_window_for_ram(64.0) == LARGE_CONTEXT_WINDOW


def test_windows_are_at_least_128k():
    for ram in (2, 4, 8, 16, 32, 128):
        assert context_window_for_ram(ram) >= 131_072


def test_estimate_tokens_is_conservative():
    assert estimate_tokens("") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 400) == 100


def test_estimate_messages_includes_per_message_overhead():
    one = estimate_messages_tokens([{"content": "a" * 400}])
    two = estimate_messages_tokens([{"content": "a" * 400}, {"content": "a" * 400}])
    assert two > one
    assert one >= 100


def test_describe_policy_mentions_the_window_and_machine():
    assert "128k" in describe_policy(8.0)
    assert "256k" in describe_policy(16.0)
    assert ">8 GB" in describe_policy(16.0)
