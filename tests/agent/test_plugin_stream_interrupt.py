"""A plugin may ASK the core to cut a stream; only the core decides whether that happens.

The ask is one flag on an object handed to stream hooks — no waiting, no callback into the agent from
the plugin's worker thread, nothing that can block the token path. Honouring it needs the user's
``plugins.stream_interrupt_requests`` setting, and the cut lands in the same hand-off the runaway guard
uses so the turn ends instead of continuing the text the plugin objected to.

The agent here is a stub carrying only the delivery mixin: the contract under test is the payload and
the flag, and constructing a real AIAgent would read the developer's own config for no extra coverage.
"""

import threading
import time
from types import SimpleNamespace
from unittest.mock import patch


class _StubAgentMixin:
    """The attributes ``StreamDeliveryMixin``'s stream-hook helpers read."""

    session_id = "test-session"
    model = "test/model"
    provider = "openrouter"
    platform = "cli"
    _current_turn_id = ""
    _api_call_count = 0
    stream_delta_callback = None


def _agent(monkeypatch, callbacks=None):
    from agent.stream_delivery import StreamDeliveryMixin

    class Stub(StreamDeliveryMixin, _StubAgentMixin):
        # ``_fire_stream_delta`` scrubs think-tags before enqueuing; identity is enough here.
        _strip_think_blocks = staticmethod(lambda text, *a, **k: text)

    if callbacks is not None:
        monkeypatch.setattr("hermes_cli.plugins.iter_hook_callbacks", callbacks)
    return Stub()


def _request():
    from agent.plugin_stream_hooks import StreamInterruptRequest

    return StreamInterruptRequest()


def test_a_request_is_a_latch_a_plugin_cannot_refire():
    request = _request()

    assert request.requested() is False
    assert request.reason() == ""
    assert request.request("degenerate loop", source="degeneracy-guard") is True
    assert request.requested() is True
    assert request.reason() == "degeneracy-guard: degenerate loop"
    # Second ask (a chatty plugin, another worker) must not rewrite why the stream was cut.
    assert request.request("something else", source="other-plugin") is False
    assert request.reason() == "degeneracy-guard: degenerate loop"


def test_a_request_never_waits_on_the_worker_thread():
    request = _request()
    done = threading.Event()

    def worker():
        request.request("asking from a worker", source="test-plugin")
        done.set()

    thread = threading.Thread(target=worker)
    thread.start()
    assert done.wait(timeout=2.0)
    thread.join(timeout=2.0)

    assert request.requested() is True


def test_module_helper_tolerates_a_missing_request():
    from agent.plugin_stream_hooks import request_stream_interrupt

    # A hook fired outside a stream hands the plugin None; that must be a no-op, not a crash.
    assert request_stream_interrupt(None, "no stream here") is False


def test_enabled_flag_defaults_off_and_reads_the_config_key(monkeypatch):
    from agent.plugin_stream_hooks import stream_interrupt_requests_enabled

    with patch("hermes_cli.config.load_config_readonly", return_value={}):
        assert stream_interrupt_requests_enabled() is False

    cfg = {"plugins": {"stream_interrupt_requests": True}}
    with patch("hermes_cli.config.load_config_readonly", return_value=cfg):
        assert stream_interrupt_requests_enabled() is True


def test_stream_hook_payload_carries_the_request_object(monkeypatch):
    from agent.plugin_stream_hooks import request_stream_interrupt

    agent = _agent(monkeypatch)
    payload = agent._stream_hook_base_payload()

    interrupt = payload["interrupt"]
    assert interrupt.requested() is False
    # The same object every hook for this response receives, so an ask reaches the core's check.
    assert payload["interrupt"] is agent._plugin_stream_interrupt()
    # The plugin side of the contract: one call, no arguments it must know about.
    assert request_stream_interrupt(interrupt, "repeating", source="test-plugin") is True
    # With the user's opt-in off, the ask is recorded but not honoured.
    monkeypatch.setattr(agent, "_plugin_stream_interrupts_enabled", lambda: False)
    assert agent._plugin_interrupt_pending() is False


def test_the_core_ignores_the_ask_unless_the_user_enabled_it(monkeypatch):
    agent = _agent(monkeypatch)
    agent._stream_interrupt_request = None
    agent._stream_interrupts_enabled = None
    interrupt = agent._plugin_stream_interrupt()
    interrupt.request("repeating", source="test-plugin")

    monkeypatch.setattr(agent, "_plugin_stream_interrupts_enabled", lambda: False)
    assert agent._plugin_interrupt_pending() is False

    monkeypatch.setattr(agent, "_plugin_stream_interrupts_enabled", lambda: True)
    assert agent._plugin_interrupt_pending() is True
    assert agent._take_plugin_interrupt_reason() == "test-plugin: repeating"


def test_each_model_response_gets_a_fresh_request(monkeypatch):
    agent = _agent(monkeypatch)
    agent._stream_interrupt_request = None
    first = agent._plugin_stream_interrupt()
    first.request("repeating", source="test-plugin")
    assert agent._plugin_stream_interrupt() is first

    agent._reset_stream_delivery_tracking()

    fresh = agent._plugin_stream_interrupt()
    assert fresh is not first
    assert fresh.requested() is False


def test_the_turn_ends_on_the_plugin_cut_instead_of_continuing(monkeypatch):
    from agent import turn_truncation
    from agent.turn_retry_state import TurnRetryState

    ended = {}

    class _StubTrunc:
        def __init__(self, **kwargs):
            ended["agent"] = kwargs["agent"]

        def end_turn(self, final_response, error=None, **_kwargs):
            ended["final_response"] = final_response
            ended["error"] = error
            return "ended"

    monkeypatch.setattr(turn_truncation, "_Trunc", _StubTrunc)
    printed = []
    agent = SimpleNamespace(log_prefix="[test] ", _vprint=lambda msg, **_kw: printed.append(msg))
    response = SimpleNamespace(_plugin_interrupt_reason="degeneracy-guard: degenerate loop")

    verdict = turn_truncation.recover_from_truncation(
        agent, response, "length", TurnRetryState(), messages=[], conversation_history=[],
        api_kwargs={}, api_call_count=1, effective_task_id="task", current_turn_user_idx=0,
        length_continue_retries=0, truncated_response_parts=[], truncated_tool_call_retries=0,
        retry_count=0, compression_attempts=0,
    )

    assert verdict == "ended"
    assert ended["error"] == "plugin_stream_interrupt"
    assert "degeneracy-guard: degenerate loop" in printed[0]


def _guard_plugin():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "fixtures" / "plugins" / "stream_interrupt_guard.py"
    spec = importlib.util.spec_from_file_location("stream_interrupt_guard", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Ctx:
    def __init__(self) -> None:
        self.hooks = {}

    def register_hook(self, name, fn) -> None:
        self.hooks[name] = fn


def test_the_example_guard_asks_to_cut_a_degenerate_stream():
    module = _guard_plugin()
    ctx = _Ctx()
    module.register(ctx)
    interrupt = _request()

    ctx.hooks["on_stream_start"]()
    for _ in range(5):
        ctx.hooks["on_stream_delta"](delta="the model is looping again ", kind="text", interrupt=interrupt)

    assert interrupt.requested() is True
    assert interrupt.reason().startswith("degeneracy-guard: ")


def test_the_example_guard_leaves_a_normal_stream_alone():
    module = _guard_plugin()
    ctx = _Ctx()
    module.register(ctx)
    interrupt = _request()

    ctx.hooks["on_stream_start"]()
    for delta in ("Here is the summary. ", "The build passed on all three jobs. ", "I opened #1443 next."):
        ctx.hooks["on_stream_delta"](delta=delta, kind="text", interrupt=interrupt)

    assert interrupt.requested() is False


def test_an_ask_from_a_hook_worker_reaches_the_core_gate(monkeypatch):
    from agent.plugin_stream_hooks import enqueue_plugin_stream_hook, shutdown_plugin_stream_hook_dispatcher
    from agent.plugin_stream_hooks import StreamInterruptRequest

    shutdown_plugin_stream_hook_dispatcher()
    module = _guard_plugin()
    ctx = _Ctx()
    module.register(ctx)
    monkeypatch.setattr(
        "hermes_cli.plugins.iter_hook_callbacks",
        lambda name: (ctx.hooks[name],) if name in ctx.hooks else (),
    )

    agent = _agent(monkeypatch)
    agent._stream_interrupt_request = StreamInterruptRequest()
    agent._stream_interrupts_enabled = True
    payload = agent._stream_hook_base_payload()

    for _ in range(5):
        enqueue_plugin_stream_hook("on_stream_delta", delta="looping again and again ", kind="text", **payload)

    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and not agent._plugin_interrupt_pending():
        time.sleep(0.01)
    shutdown_plugin_stream_hook_dispatcher()

    assert agent._plugin_interrupt_pending() is True
    assert agent._take_plugin_interrupt_reason().startswith("degeneracy-guard: ")
