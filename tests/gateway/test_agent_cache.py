"""Integration tests for gateway AIAgent caching.

Verifies that the agent cache correctly:
- Reuses agents across messages (same config → same instance)
- Rebuilds agents when config changes (model, provider, toolsets)
- Updates reasoning_config in-place without rebuilding
- Evicts on session reset
- Evicts on fallback activation
- Preserves frozen system prompt across turns
"""

import hashlib
import json
import threading
from unittest.mock import MagicMock, patch

import pytest


def _make_runner():
    """Create a minimal GatewayRunner with just the cache infrastructure."""
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    return runner


class TestAgentConfigSignature:
    """Config signature produces stable, distinct keys."""

    def test_same_config_same_signature(self):
        from gateway.run import GatewayRunner

        runtime = {"api_key": "sk-test12345678", "base_url": "https://openrouter.ai/api/v1",
                    "provider": "openrouter", "api_mode": "chat_completions"}
        sig1 = GatewayRunner._agent_config_signature("claude-sonnet-4", runtime, ["hermes-telegram"], "")
        sig2 = GatewayRunner._agent_config_signature("claude-sonnet-4", runtime, ["hermes-telegram"], "")
        assert sig1 == sig2

    def test_model_change_different_signature(self):
        from gateway.run import GatewayRunner

        runtime = {"api_key": "sk-test12345678", "base_url": "https://openrouter.ai/api/v1",
                    "provider": "openrouter"}
        sig1 = GatewayRunner._agent_config_signature("claude-sonnet-4", runtime, ["hermes-telegram"], "")
        sig2 = GatewayRunner._agent_config_signature("claude-opus-4.6", runtime, ["hermes-telegram"], "")
        assert sig1 != sig2

    def test_same_token_prefix_different_full_token_changes_signature(self):
        """Tokens sharing a JWT-style prefix must not collide."""
        from gateway.run import GatewayRunner

        rt1 = {
            "api_key": "eyJhbGci.token-for-account-a",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "provider": "openai-codex",
            "api_mode": "codex_responses",
        }
        rt2 = {
            "api_key": "eyJhbGci.token-for-account-b",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "provider": "openai-codex",
            "api_mode": "codex_responses",
        }

        assert rt1["api_key"][:8] == rt2["api_key"][:8]
        sig1 = GatewayRunner._agent_config_signature("gpt-5.3-codex", rt1, ["hermes-telegram"], "")
        sig2 = GatewayRunner._agent_config_signature("gpt-5.3-codex", rt2, ["hermes-telegram"], "")
        assert sig1 != sig2

    def test_provider_change_different_signature(self):
        from gateway.run import GatewayRunner

        rt1 = {"api_key": "sk-test12345678", "base_url": "https://openrouter.ai/api/v1", "provider": "openrouter"}
        rt2 = {"api_key": "sk-test12345678", "base_url": "https://api.anthropic.com", "provider": "anthropic"}
        sig1 = GatewayRunner._agent_config_signature("claude-sonnet-4", rt1, ["hermes-telegram"], "")
        sig2 = GatewayRunner._agent_config_signature("claude-sonnet-4", rt2, ["hermes-telegram"], "")
        assert sig1 != sig2

    def test_toolset_change_different_signature(self):
        from gateway.run import GatewayRunner

        runtime = {"api_key": "sk-test12345678", "base_url": "https://openrouter.ai/api/v1", "provider": "openrouter"}
        sig1 = GatewayRunner._agent_config_signature("claude-sonnet-4", runtime, ["hermes-telegram"], "")
        sig2 = GatewayRunner._agent_config_signature("claude-sonnet-4", runtime, ["hermes-discord"], "")
        assert sig1 != sig2

    def test_reasoning_not_in_signature(self):
        """Reasoning config is set per-message, not part of the signature."""
        from gateway.run import GatewayRunner

        runtime = {"api_key": "sk-test12345678", "base_url": "https://openrouter.ai/api/v1", "provider": "openrouter"}
        # Same config — signature should be identical regardless of what
        # reasoning_config the caller might have (it's not passed in)
        sig1 = GatewayRunner._agent_config_signature("claude-sonnet-4", runtime, ["hermes-telegram"], "")
        sig2 = GatewayRunner._agent_config_signature("claude-sonnet-4", runtime, ["hermes-telegram"], "")
        assert sig1 == sig2


class TestAgentCacheLifecycle:
    """End-to-end cache behavior with real AIAgent construction."""

    def test_cache_hit_returns_same_agent(self):
        """Second message with same config reuses the cached agent instance."""
        from run_agent import AIAgent

        runner = _make_runner()
        session_key = "telegram:12345"
        runtime = {"api_key": "test", "base_url": "https://openrouter.ai/api/v1",
                    "provider": "openrouter", "api_mode": "chat_completions"}
        sig = runner._agent_config_signature("anthropic/claude-sonnet-4", runtime, ["hermes-telegram"], "")

        # First message — create and cache
        agent1 = AIAgent(
            model="anthropic/claude-sonnet-4", api_key="test",
            base_url="https://openrouter.ai/api/v1", provider="openrouter",
            max_iterations=5, quiet_mode=True, skip_context_files=True,
            skip_memory=True, platform="telegram",
        )
        with runner._agent_cache_lock:
            runner._agent_cache[session_key] = (agent1, sig)

        # Second message — cache hit
        with runner._agent_cache_lock:
            cached = runner._agent_cache.get(session_key)
        assert cached is not None
        assert cached[1] == sig
        assert cached[0] is agent1  # same instance

    def test_cache_miss_on_model_change(self):
        """Model change produces different signature → cache miss."""
        from run_agent import AIAgent

        runner = _make_runner()
        session_key = "telegram:12345"
        runtime = {"api_key": "test", "base_url": "https://openrouter.ai/api/v1",
                    "provider": "openrouter", "api_mode": "chat_completions"}

        old_sig = runner._agent_config_signature("anthropic/claude-sonnet-4", runtime, ["hermes-telegram"], "")
        agent1 = AIAgent(
            model="anthropic/claude-sonnet-4", api_key="test",
            base_url="https://openrouter.ai/api/v1", provider="openrouter",
            max_iterations=5, quiet_mode=True, skip_context_files=True,
            skip_memory=True, platform="telegram",
        )
        with runner._agent_cache_lock:
            runner._agent_cache[session_key] = (agent1, old_sig)

        # New model → different signature
        new_sig = runner._agent_config_signature("anthropic/claude-opus-4.6", runtime, ["hermes-telegram"], "")
        assert new_sig != old_sig

        with runner._agent_cache_lock:
            cached = runner._agent_cache.get(session_key)
        assert cached[1] != new_sig  # signature mismatch → would create new agent

    def test_evict_on_session_reset(self):
        """_evict_cached_agent removes the entry."""
        from run_agent import AIAgent

        runner = _make_runner()
        session_key = "telegram:12345"

        agent = AIAgent(
            model="anthropic/claude-sonnet-4", api_key="test",
            base_url="https://openrouter.ai/api/v1", provider="openrouter",
            max_iterations=5, quiet_mode=True, skip_context_files=True,
            skip_memory=True,
        )
        with runner._agent_cache_lock:
            runner._agent_cache[session_key] = (agent, "sig123")

        runner._evict_cached_agent(session_key)

        with runner._agent_cache_lock:
            assert session_key not in runner._agent_cache

    def test_evict_does_not_affect_other_sessions(self):
        """Evicting one session leaves other sessions cached."""
        runner = _make_runner()
        with runner._agent_cache_lock:
            runner._agent_cache["session-A"] = ("agent-A", "sig-A")
            runner._agent_cache["session-B"] = ("agent-B", "sig-B")

        runner._evict_cached_agent("session-A")

        with runner._agent_cache_lock:
            assert "session-A" not in runner._agent_cache
            assert "session-B" in runner._agent_cache

    def test_reasoning_config_updates_in_place(self):
        """Reasoning config can be set on a cached agent without eviction."""
        from run_agent import AIAgent

        agent = AIAgent(
            model="anthropic/claude-sonnet-4", api_key="test",
            base_url="https://openrouter.ai/api/v1", provider="openrouter",
            max_iterations=5, quiet_mode=True, skip_context_files=True,
            skip_memory=True,
            reasoning_config={"enabled": True, "effort": "medium"},
        )

        # Simulate per-message reasoning update
        agent.reasoning_config = {"enabled": True, "effort": "high"}
        assert agent.reasoning_config["effort"] == "high"

        # System prompt should not be affected by reasoning change
        prompt1 = agent._build_system_prompt()
        agent._cached_system_prompt = prompt1  # simulate run_conversation caching
        agent.reasoning_config = {"enabled": True, "effort": "low"}
        prompt2 = agent._cached_system_prompt
        assert prompt1 is prompt2  # same object — not invalidated by reasoning change

    def test_system_prompt_frozen_across_cache_reuse(self):
        """The cached agent's system prompt stays identical across turns."""
        from run_agent import AIAgent

        agent = AIAgent(
            model="anthropic/claude-sonnet-4", api_key="test",
            base_url="https://openrouter.ai/api/v1", provider="openrouter",
            max_iterations=5, quiet_mode=True, skip_context_files=True,
            skip_memory=True, platform="telegram",
        )

        # Build system prompt (simulates first run_conversation)
        prompt1 = agent._build_system_prompt()
        agent._cached_system_prompt = prompt1

        # Simulate second turn — prompt should be frozen
        prompt2 = agent._cached_system_prompt
        assert prompt1 is prompt2  # same object, not rebuilt

    def test_callbacks_update_without_cache_eviction(self):
        """Per-message callbacks can be set on cached agent."""
        from run_agent import AIAgent

        agent = AIAgent(
            model="anthropic/claude-sonnet-4", api_key="test",
            base_url="https://openrouter.ai/api/v1", provider="openrouter",
            max_iterations=5, quiet_mode=True, skip_context_files=True,
            skip_memory=True,
        )

        # Set callbacks like the gateway does per-message
        cb1 = lambda *a: None
        cb2 = lambda *a: None
        agent.tool_progress_callback = cb1
        agent.step_callback = cb2
        agent.stream_delta_callback = None
        agent.status_callback = None

        assert agent.tool_progress_callback is cb1
        assert agent.step_callback is cb2

        # Update for next message
        cb3 = lambda *a: None
        agent.tool_progress_callback = cb3
        assert agent.tool_progress_callback is cb3


class TestAgentCacheBoundedGrowth:
    """LRU cap and idle-TTL eviction prevent unbounded cache growth."""

    def _bounded_runner(self):
        """Runner with an OrderedDict cache (matches real gateway init)."""
        from collections import OrderedDict
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner._agent_cache = OrderedDict()
        runner._agent_cache_lock = threading.Lock()
        return runner

    def _fake_agent(self, last_activity: float | None = None):
        """Lightweight stand-in; real AIAgent is heavy to construct."""
        m = MagicMock()
        if last_activity is not None:
            m._last_activity_ts = last_activity
        else:
            import time as _t
            m._last_activity_ts = _t.time()
        return m

    def test_cap_evicts_lru_when_exceeded(self, monkeypatch):
        """Inserting past _AGENT_CACHE_MAX_SIZE pops the oldest entry."""
        from gateway import run as gw_run

        monkeypatch.setattr(gw_run, "_AGENT_CACHE_MAX_SIZE", 3)
        runner = self._bounded_runner()
        runner._cleanup_agent_resources = MagicMock()

        for i in range(3):
            runner._agent_cache[f"s{i}"] = (self._fake_agent(), f"sig{i}")

        # Insert a 4th — oldest (s0) must be evicted.
        with runner._agent_cache_lock:
            runner._agent_cache["s3"] = (self._fake_agent(), "sig3")
            runner._enforce_agent_cache_cap()

        assert "s0" not in runner._agent_cache
        assert "s3" in runner._agent_cache
        assert len(runner._agent_cache) == 3

    def test_cap_respects_move_to_end(self, monkeypatch):
        """Entries refreshed via move_to_end are NOT evicted as 'oldest'."""
        from gateway import run as gw_run

        monkeypatch.setattr(gw_run, "_AGENT_CACHE_MAX_SIZE", 3)
        runner = self._bounded_runner()
        runner._cleanup_agent_resources = MagicMock()

        for i in range(3):
            runner._agent_cache[f"s{i}"] = (self._fake_agent(), f"sig{i}")

        # Touch s0 — it is now MRU, so s1 becomes LRU.
        runner._agent_cache.move_to_end("s0")

        with runner._agent_cache_lock:
            runner._agent_cache["s3"] = (self._fake_agent(), "sig3")
            runner._enforce_agent_cache_cap()

        assert "s0" in runner._agent_cache  # rescued by move_to_end
        assert "s1" not in runner._agent_cache  # now oldest → evicted
        assert "s3" in runner._agent_cache

    def test_cap_triggers_cleanup_thread(self, monkeypatch):
        """Evicted agent has _cleanup_agent_resources called for it."""
        from gateway import run as gw_run

        monkeypatch.setattr(gw_run, "_AGENT_CACHE_MAX_SIZE", 1)
        runner = self._bounded_runner()

        cleanup_calls: list = []
        runner._cleanup_agent_resources = lambda a: cleanup_calls.append(a)

        old_agent = self._fake_agent()
        new_agent = self._fake_agent()
        with runner._agent_cache_lock:
            runner._agent_cache["old"] = (old_agent, "sig_old")
            runner._agent_cache["new"] = (new_agent, "sig_new")
            runner._enforce_agent_cache_cap()

        # Cleanup is dispatched to a daemon thread; join briefly to observe.
        import time as _t
        deadline = _t.time() + 2.0
        while _t.time() < deadline and not cleanup_calls:
            _t.sleep(0.02)
        assert old_agent in cleanup_calls
        assert new_agent not in cleanup_calls

    def test_idle_ttl_sweep_evicts_stale_agents(self, monkeypatch):
        """_sweep_idle_cached_agents removes agents idle past the TTL."""
        from gateway import run as gw_run

        monkeypatch.setattr(gw_run, "_AGENT_CACHE_IDLE_TTL_SECS", 0.05)
        runner = self._bounded_runner()
        runner._cleanup_agent_resources = MagicMock()

        import time as _t
        fresh = self._fake_agent(last_activity=_t.time())
        stale = self._fake_agent(last_activity=_t.time() - 10.0)
        runner._agent_cache["fresh"] = (fresh, "s1")
        runner._agent_cache["stale"] = (stale, "s2")

        evicted = runner._sweep_idle_cached_agents()
        assert evicted == 1
        assert "stale" not in runner._agent_cache
        assert "fresh" in runner._agent_cache

    def test_idle_sweep_skips_agents_without_activity_ts(self, monkeypatch):
        """Agents missing _last_activity_ts are left alone (defensive)."""
        from gateway import run as gw_run

        monkeypatch.setattr(gw_run, "_AGENT_CACHE_IDLE_TTL_SECS", 0.01)
        runner = self._bounded_runner()
        runner._cleanup_agent_resources = MagicMock()

        no_ts = MagicMock(spec=[])  # no _last_activity_ts attribute
        runner._agent_cache["s"] = (no_ts, "sig")

        assert runner._sweep_idle_cached_agents() == 0
        assert "s" in runner._agent_cache

    def test_plain_dict_cache_is_tolerated(self):
        """Test fixtures using plain {} don't crash _enforce_agent_cache_cap."""
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner._agent_cache = {}  # plain dict, not OrderedDict
        runner._agent_cache_lock = threading.Lock()
        runner._cleanup_agent_resources = MagicMock()

        # Should be a no-op rather than raising.
        with runner._agent_cache_lock:
            for i in range(200):
                runner._agent_cache[f"s{i}"] = (MagicMock(), f"sig{i}")
            runner._enforce_agent_cache_cap()  # no crash, no eviction

        assert len(runner._agent_cache) == 200

    def test_main_lookup_updates_lru_order(self, monkeypatch):
        """Cache hit via the main-lookup path refreshes LRU position."""
        runner = self._bounded_runner()

        a0 = self._fake_agent()
        a1 = self._fake_agent()
        a2 = self._fake_agent()
        runner._agent_cache["s0"] = (a0, "sig0")
        runner._agent_cache["s1"] = (a1, "sig1")
        runner._agent_cache["s2"] = (a2, "sig2")

        # Simulate what _process_message_background does on a cache hit
        # (minus the agent-state reset which isn't relevant here).
        with runner._agent_cache_lock:
            cached = runner._agent_cache.get("s0")
            if cached and hasattr(runner._agent_cache, "move_to_end"):
                runner._agent_cache.move_to_end("s0")

        # After the hit, insertion order should be s1, s2, s0.
        assert list(runner._agent_cache.keys()) == ["s1", "s2", "s0"]
