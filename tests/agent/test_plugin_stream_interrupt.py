"""A plugin may ASK the core to cut a stream; only the core decides whether that happens.

The ask is one flag on an object handed to stream hooks — no waiting, no callback into the agent from
the plugin's worker thread, nothing that can block the token path. Honouring it needs the user's
``plugins.stream_interrupt_requests`` setting, and the cut lands in the same hand-off the runaway guard
uses so the turn ends instead of continuing the text the plugin objected to.

The agent here is a stub carrying only the delivery mixin: the contract under test is the payload and
the flag, and constructing a real AIAgent would read the developer's own config for no extra coverage.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import patch

import pytest


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
    assert ended["error"].startswith("A plugin asked the core to cut this stream")
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


# --- channel level: the wire's stream loop really stops -------------------------------
#
# The tests above cover the latch and the turn's ending. These drive both wires end to end
# with a local endpoint, so the middle link is covered too: the loop asks the question between
# chunks, and stops where the plugin asked instead of reading the rest of the stream.

_CHUNK_TOTAL = 120
_CHUNK_GATE_AT = 8   # the stream pauses here until the hook has had its say
_CHUNKS = [f"item {n}: this backlog entry has its own wording, its own owner and its own date.\n"
           for n in range(_CHUNK_TOTAL)]
_FULL_TEXT = "".join(_CHUNKS)


class _WireEndpoint:
    """Local SSE endpoint for both wires. It holds the stream open at ``_CHUNK_GATE_AT`` until the
    hook has asked, so the ask always lands mid-stream: the test never races a clock."""

    def __init__(self) -> None:
        self.requests = 0
        self.gate = threading.Event()
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):  # model metadata probes
                body = json.dumps({"object": "list", "data": [
                    {"id": "fixture-local", "object": "model", "context_length": 131072}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                endpoint.requests += 1
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                frames = endpoint._anthropic_frames() if self.path.endswith("/messages") else endpoint._chat_frames()
                try:
                    for frame in frames:
                        self.wfile.write(frame.encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass  # the core closed the stream early, which is the point of the cut test

            def log_message(self, *args):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self._server.server_port}/v1"

    def _pause_for_the_ask(self, index: int) -> None:
        """Block at the gate until the plugin has spoken (bounded, so a broken hook cannot hang CI)."""
        if index + 1 == _CHUNK_GATE_AT:
            self.gate.wait(timeout=5.0)

    def _chat_frames(self):
        for index, chunk in enumerate(_CHUNKS):
            yield "data: " + json.dumps({
                "id": "chatcmpl-fixture", "object": "chat.completion.chunk", "created": 0,
                "model": "fixture-local",
                "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
            }) + "\n\n"
            self._pause_for_the_ask(index)
        yield "data: " + json.dumps({
            "id": "chatcmpl-fixture", "object": "chat.completion.chunk", "created": 0,
            "model": "fixture-local",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }) + "\n\n"
        yield "data: [DONE]\n\n"

    def _anthropic_frames(self):
        def event(kind, **payload):
            return f"event: {kind}\ndata: {json.dumps({'type': kind, **payload})}\n\n"

        yield event("message_start", message={
            "id": "msg_fixture", "type": "message", "role": "assistant", "model": "fixture-local",
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 0}})
        yield event("content_block_start", index=0, content_block={"type": "text", "text": ""})
        for position, chunk in enumerate(_CHUNKS):
            yield event("content_block_delta", index=0, delta={"type": "text_delta", "text": chunk})
            self._pause_for_the_ask(position)
        yield event("content_block_stop", index=0)
        yield event("message_delta", delta={"stop_reason": "end_turn", "stop_sequence": None},
                    usage={"output_tokens": 1})
        yield event("message_stop")

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def _ask_at(nth: int, server: "_WireEndpoint"):
    """A stream hook that asks for a cut on its nth text delta, then lets the stream continue."""
    seen = {"deltas": 0}

    def on_stream_delta(delta=None, kind=None, interrupt=None, **_kwargs):
        if kind != "text":
            return
        seen["deltas"] += 1
        if seen["deltas"] == nth and interrupt is not None:
            interrupt.request("the same block is repeating", source="test-guard")
            server.gate.set()

    return on_stream_delta, seen


def _run_wire_turn(monkeypatch, server, api_mode, *, enabled):
    from run_agent import AIAgent

    agent = AIAgent(
        model="fixture-local", provider="custom", base_url=server.url, api_key="fixture",
        api_mode=api_mode, quiet_mode=True, skip_memory=True, skip_context_files=True,
        enabled_toolsets=[], max_iterations=1,
    )
    delivered = {"text": 0}

    def count(text):
        delivered["text"] += 1
        return True

    agent.stream_delta_callback = count
    # The config key is read per response, so patching it after construction exercises the real
    # lookup without handing a fake config to the agent's own setup.
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"plugins": {"stream_interrupt_requests": enabled}},
    )
    outcome: dict = {}

    def run():
        try:
            outcome["result"] = agent.run_conversation("Summarize the backlog.")
        except BaseException as exc:  # surfaced below on the test thread
            outcome["error"] = exc

    turn = threading.Thread(target=run, daemon=True)
    turn.start()
    turn.join(60)
    if turn.is_alive():
        agent.interrupt("test bound reached")
        turn.join(10)
        pytest.fail("turn still streaming after 60s")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"], delivered["text"]


@pytest.fixture()
def wire(monkeypatch):
    from agent.plugin_stream_hooks import shutdown_plugin_stream_hook_dispatcher

    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    shutdown_plugin_stream_hook_dispatcher()
    servers: list[_WireEndpoint] = []

    def start(hook_at: int | None = None) -> tuple[_WireEndpoint, dict]:
        server = _WireEndpoint()
        servers.append(server)
        seen = {}
        if hook_at is not None:
            callback, seen = _ask_at(hook_at, server)
            monkeypatch.setattr("hermes_cli.plugins.iter_hook_callbacks",
                                lambda name: (callback,) if name == "on_stream_delta" else ())
        return server, seen

    yield start
    shutdown_plugin_stream_hook_dispatcher()
    for server in servers:
        server.close()


@pytest.mark.parametrize("api_mode", ["chat_completions", "anthropic_messages"])
def test_the_stream_loop_stops_where_the_plugin_asked(monkeypatch, wire, api_mode):
    server, _seen = wire(hook_at=8)

    result, delivered = _run_wire_turn(monkeypatch, server, api_mode, enabled=True)

    assert (result["completed"], result["failure_reason"]) == (False, "truncated")
    assert "A Plugin Asked to Cut the Stream" in result["final_response"]
    # The loop stopped near the ask instead of reading all 120 chunks: that is the link these
    # tests exist for. Slack covers deltas already in flight when the ask landed.
    assert delivered < _CHUNK_TOTAL // 2
    assert server.requests == 1
    assert _CHUNKS[-1].strip() not in result["final_response"]


@pytest.mark.parametrize("api_mode", ["chat_completions", "anthropic_messages"])
def test_the_same_ask_is_recorded_and_ignored_when_the_setting_is_off(monkeypatch, wire, api_mode):
    server, seen = wire(hook_at=8)

    result, delivered = _run_wire_turn(monkeypatch, server, api_mode, enabled=False)

    assert result["completed"] is True
    assert result["final_response"].strip() == _FULL_TEXT.strip()
    assert delivered == _CHUNK_TOTAL
    assert seen["deltas"] == _CHUNK_TOTAL  # the plugin still saw everything it asked about
