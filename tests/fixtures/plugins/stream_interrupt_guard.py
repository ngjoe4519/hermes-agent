"""Smallest useful consumer of the stream interrupt request: stop a response that repeats itself.

This mirrors what a degeneracy guard needs — it sees deltas only, so it cannot close the provider
connection or stop the loop; all it can do is say "this one is going wrong" and let the core decide.
"""

_WINDOW_WORDS = 240
_REPEATS = 4          # the same block this many times in a row is not prose
_MAX_PERIOD = 8       # longest block we bother to test (words)


def _words(text: str) -> list[str]:
    return text.split()


def _repeats(words: list[str]) -> bool:
    """True when the tail ends in ``_REPEATS`` copies of the same 1..8 word block."""
    total = len(words)
    for period in range(1, _MAX_PERIOD + 1):
        if total < period * _REPEATS:
            continue
        blocks = [words[total - (index + 1) * period: total - index * period]
                  for index in range(_REPEATS)]
        if all(block == blocks[0] for block in blocks):
            return True
    return False


class DegeneracyGuard:
    """Accumulates a small window of text and asks for a cut once the same block repeats."""

    def __init__(self) -> None:
        self._tail = ""

    def reset(self) -> None:
        self._tail = ""

    def observe(self, delta: str) -> str | None:
        """Return a reason the moment the tail degenerates, otherwise ``None``."""
        self._tail = " ".join(_words(self._tail + " " + delta)[-_WINDOW_WORDS:])
        if _repeats(_words(self._tail)):
            return "the same phrase repeated four times in a row"
        return None


def register(ctx) -> None:
    guard = DegeneracyGuard()

    def on_stream_start(**_kwargs) -> None:
        guard.reset()

    def on_stream_delta(delta: str, kind: str, interrupt=None, **_kwargs) -> None:
        if kind != "text":
            return
        reason = guard.observe(delta)
        if reason and interrupt is not None:
            interrupt.request(reason, source="degeneracy-guard")

    ctx.register_hook("on_stream_start", on_stream_start)
    ctx.register_hook("on_stream_delta", on_stream_delta)
