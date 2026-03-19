"""Tests for WorkerLoop — DB-polling worker for background execution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from dendrite.runtime.worker import WorkerLoop

# ------------------------------------------------------------------
# Mock registry + state store
# ------------------------------------------------------------------


@dataclass
class _RunRecord:
    id: str = ""
    agent_name: str = "TestAgent"
    status: str = "pending"
    input_data: dict[str, Any] | None = None
    output_data: dict[str, Any] | None = None
    answer: str | None = None
    error: str | None = None
    iteration_count: int = 0
    model: str | None = None
    strategy: str | None = None
    parent_run_id: str | None = None
    delegation_level: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float | None = None
    meta: dict[str, Any] | None = None
    retry_count: int = 0
    created_at: Any = None
    updated_at: Any = None


@dataclass
class _EventRecord:
    id: str = ""
    event_type: str = ""
    sequence_index: int = 0
    iteration_index: int = 0
    correlation_id: str | None = None
    data: dict[str, Any] | None = None
    created_at: Any = None


class WorkerMockStore:
    """Mock state store for worker tests."""

    def __init__(self) -> None:
        self.runs: dict[str, _RunRecord] = {}
        self.stale_run_ids: list[str] = []
        self.reclaimed: list[str] = []
        self.claimed: list[str] = []
        # Track calls for assertion
        self.list_runs_calls: list[dict[str, Any]] = []

    async def list_runs(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        tenant_id: str | None = None,
        status: str | None = None,
    ) -> list[_RunRecord]:
        self.list_runs_calls.append({"limit": limit, "status": status})
        return [r for r in self.runs.values() if (status is None or r.status == status)][:limit]

    async def find_stale_runs(self, threshold_seconds: int) -> list[str]:
        return list(self.stale_run_ids)

    async def reclaim_stale_run(self, run_id: str) -> bool:
        self.reclaimed.append(run_id)
        if run_id in self.runs:
            run = self.runs[run_id]
            run.status = "pending"
            run.retry_count += 1
            return True
        return False

    # Lease methods needed by execute_pending_run / recover_run
    async def get_run(self, run_id: str) -> _RunRecord | None:
        return self.runs.get(run_id)

    async def claim_run(self, run_id: str, executor_id: str) -> str | None:
        run = self.runs.get(run_id)
        if run and run.status == "pending":
            run.status = "running"
            self.claimed.append(run_id)
            return "nonce-" + run_id
        return None

    async def renew_heartbeat(self, run_id: str, lease_nonce: str) -> bool:
        return True

    async def release_lease(self, run_id: str, lease_nonce: str) -> None:
        pass

    async def get_run_events(self, run_id: str) -> list[_EventRecord]:
        return []

    async def get_traces(self, run_id: str) -> list[Any]:
        return []

    async def save_trace(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def save_tool_call(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def save_usage(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def save_llm_interaction(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def save_run_event(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def finalize_run(self, run_id: str, **kwargs: Any) -> bool:
        if run_id in self.runs:
            self.runs[run_id].status = kwargs.get("status", "success")
        return True

    async def pause_run(self, run_id: str, **kwargs: Any) -> bool:
        if run_id in self.runs:
            self.runs[run_id].status = kwargs.get("status", "waiting_client_tool")
        return True

    async def get_pause_state(self, run_id: str) -> dict[str, Any] | None:
        return None

    async def get_llm_interactions(self, run_id: str) -> list[Any]:
        return []


class MockAgentConfig:
    """Minimal HostedAgentConfig-like for tests."""

    def __init__(self, agent: Any, provider_factory: Any) -> None:
        self.agent = agent
        self.provider_factory = provider_factory
        self.strategy_factory = None
        self.loop_factory = None
        self.redact = None


class MockRegistry:
    """Minimal AgentRegistry-like for tests."""

    def __init__(self, configs: dict[str, MockAgentConfig] | None = None) -> None:
        self._configs = configs or {}

    def get(self, agent_name: str) -> MockAgentConfig:
        if agent_name not in self._configs:
            raise KeyError(f"Agent '{agent_name}' not registered")
        return self._configs[agent_name]


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestPollOnce:
    """Tests for _poll_once — finding and spawning pending runs."""

    async def test_picks_up_fresh_pending_run(self) -> None:
        """Worker finds a pending run and spawns a task for it."""
        store = WorkerMockStore()
        store.runs["r1"] = _RunRecord(
            id="r1",
            agent_name="TestAgent",
            status="pending",
            input_data={"input": "hello"},
            retry_count=0,
        )

        from dendrite.agent import Agent

        agent = Agent(model="mock", prompt="Test.", tools=[])

        from dendrite.llm import MockLLM
        from dendrite.types import LLMResponse

        registry = MockRegistry(
            {
                "TestAgent": MockAgentConfig(
                    agent=agent,
                    provider_factory=lambda: MockLLM([LLMResponse(text="done")]),
                )
            }
        )

        worker = WorkerLoop(
            state_store=store,  # type: ignore[arg-type]
            registry=registry,  # type: ignore[arg-type]
            max_concurrent=5,
        )

        # Patch _run_task to verify it's called
        with patch.object(worker, "_run_task", new_callable=AsyncMock):
            spawned = await worker._poll_once()

        assert spawned == 1
        # Task was created in _active_tasks
        # (we patched _run_task so it won't actually run)

    async def test_recovers_reclaimed_pending_run(self) -> None:
        """Pending run with retry_count > 0 triggers recover path."""
        store = WorkerMockStore()
        store.runs["r1"] = _RunRecord(
            id="r1",
            agent_name="TestAgent",
            status="pending",
            input_data={"input": "hello"},
            retry_count=1,
        )

        from dendrite.agent import Agent

        agent = Agent(model="mock", prompt="Test.", tools=[])

        from dendrite.llm import MockLLM
        from dendrite.types import LLMResponse

        registry = MockRegistry(
            {
                "TestAgent": MockAgentConfig(
                    agent=agent,
                    provider_factory=lambda: MockLLM([LLMResponse(text="done")]),
                )
            }
        )

        worker = WorkerLoop(
            state_store=store,  # type: ignore[arg-type]
            registry=registry,  # type: ignore[arg-type]
        )

        # Track which path was taken
        calls: list[str] = []

        async def tracking_run_task(run_id: str, agent_name: str, *, recover: bool) -> None:
            calls.append(f"{'recover' if recover else 'execute'}:{run_id}")

        worker._run_task = tracking_run_task  # type: ignore[assignment]
        spawned = await worker._poll_once()

        assert spawned == 1
        # Need to let the spawned tasks run
        await asyncio.sleep(0.01)
        assert "recover:r1" in calls

    async def test_respects_max_concurrent(self) -> None:
        """Worker only spawns up to max_concurrent - active_count tasks."""
        store = WorkerMockStore()
        for i in range(5):
            store.runs[f"r{i}"] = _RunRecord(
                id=f"r{i}",
                agent_name="TestAgent",
                status="pending",
                input_data={"input": f"input-{i}"},
            )

        from dendrite.agent import Agent

        agent = Agent(model="mock", prompt="Test.", tools=[])

        from dendrite.llm import MockLLM
        from dendrite.types import LLMResponse

        registry = MockRegistry(
            {
                "TestAgent": MockAgentConfig(
                    agent=agent,
                    provider_factory=lambda: MockLLM([LLMResponse(text="done")]),
                )
            }
        )

        worker = WorkerLoop(
            state_store=store,  # type: ignore[arg-type]
            registry=registry,  # type: ignore[arg-type]
            max_concurrent=2,
        )

        # Make _run_task a no-op so tasks stay "active"
        async def slow_task(run_id: str, agent_name: str, *, recover: bool) -> None:
            await asyncio.sleep(10)  # won't actually wait — test controls time

        worker._run_task = slow_task  # type: ignore[assignment]

        spawned = await worker._poll_once()
        assert spawned == 2  # only 2, not 5
        assert len(worker._active_tasks) == 2

        # Verify list_runs was called with limit=2
        assert store.list_runs_calls[-1]["limit"] == 2

    async def test_skips_unregistered_agent(self) -> None:
        """Runs for agents not in the registry are skipped."""
        store = WorkerMockStore()
        store.runs["r1"] = _RunRecord(
            id="r1",
            agent_name="UnknownAgent",
            status="pending",
            input_data={"input": "hello"},
        )

        registry = MockRegistry({})  # no agents registered
        worker = WorkerLoop(
            state_store=store,  # type: ignore[arg-type]
            registry=registry,  # type: ignore[arg-type]
        )

        spawned = await worker._poll_once()
        assert spawned == 0

    async def test_claim_race_is_harmless(self) -> None:
        """If claim fails (another worker got it), task handles it gracefully."""
        store = WorkerMockStore()
        store.runs["r1"] = _RunRecord(
            id="r1",
            agent_name="TestAgent",
            status="pending",
            input_data={"input": "hello"},
        )

        # Make claim_run fail (simulate race)
        async def failing_claim(run_id: str, executor_id: str) -> str | None:
            return None  # another worker claimed it

        store.claim_run = failing_claim  # type: ignore[assignment]

        from dendrite.agent import Agent

        agent = Agent(model="mock", prompt="Test.", tools=[])

        from dendrite.llm import MockLLM
        from dendrite.types import LLMResponse

        registry = MockRegistry(
            {
                "TestAgent": MockAgentConfig(
                    agent=agent,
                    provider_factory=lambda: MockLLM([LLMResponse(text="done")]),
                )
            }
        )

        worker = WorkerLoop(
            state_store=store,  # type: ignore[arg-type]
            registry=registry,  # type: ignore[arg-type]
        )

        # Poll will spawn a task, but the task should handle the claim failure
        spawned = await worker._poll_once()
        assert spawned == 1

        # Let the task run and fail gracefully
        await asyncio.sleep(0.1)
        # Task should have cleaned itself up — no crash
        assert worker.active_count == 0


class TestSweepOnce:
    """Tests for _sweep_once — finding and reclaiming stale runs."""

    async def test_sweeps_stale_runs(self) -> None:
        """Sweeper finds and reclaims stale runs."""
        store = WorkerMockStore()
        store.runs["r1"] = _RunRecord(id="r1", status="running")
        store.stale_run_ids = ["r1"]

        registry = MockRegistry()
        worker = WorkerLoop(
            state_store=store,  # type: ignore[arg-type]
            registry=registry,  # type: ignore[arg-type]
        )

        reclaimed = await worker._sweep_once()
        assert reclaimed == 1
        assert "r1" in store.reclaimed
        # Run should now be pending (reclaimed)
        assert store.runs["r1"].status == "pending"

    async def test_does_not_sweep_waiting_runs(self) -> None:
        """Waiting runs are never returned by find_stale_runs (invariant 5).

        This test verifies the worker does not attempt to reclaim them
        even if they somehow appear. The real invariant is enforced by
        find_stale_runs() only querying status='running'.
        """
        store = WorkerMockStore()
        store.runs["r_wait"] = _RunRecord(id="r_wait", status="waiting_client_tool")
        # Simulate: find_stale_runs correctly returns empty
        store.stale_run_ids = []

        registry = MockRegistry()
        worker = WorkerLoop(
            state_store=store,  # type: ignore[arg-type]
            registry=registry,  # type: ignore[arg-type]
        )

        reclaimed = await worker._sweep_once()
        assert reclaimed == 0
        assert store.runs["r_wait"].status == "waiting_client_tool"

    async def test_does_not_sweep_terminal_runs(self) -> None:
        """Terminal runs are never returned by find_stale_runs (invariant 6)."""
        store = WorkerMockStore()
        store.runs["r_done"] = _RunRecord(id="r_done", status="success")
        store.stale_run_ids = []  # find_stale_runs excludes terminal

        registry = MockRegistry()
        worker = WorkerLoop(
            state_store=store,  # type: ignore[arg-type]
            registry=registry,  # type: ignore[arg-type]
        )

        reclaimed = await worker._sweep_once()
        assert reclaimed == 0


class TestShutdown:
    """Tests for graceful shutdown behavior."""

    async def test_shutdown_stops_new_claims_drains_active(self) -> None:
        """stop() prevents new polls and waits for active tasks."""
        store = WorkerMockStore()
        registry = MockRegistry()

        worker = WorkerLoop(
            state_store=store,  # type: ignore[arg-type]
            registry=registry,  # type: ignore[arg-type]
            poll_interval=0.05,
            sweep_interval=0.05,
        )

        # Add a "running" task that completes quickly
        completed = asyncio.Event()

        async def quick_task() -> None:
            await asyncio.sleep(0.05)
            completed.set()

        worker._active_tasks["r_active"] = asyncio.create_task(quick_task())

        # Start worker in background
        worker_task = asyncio.create_task(worker.start())

        # Give the worker a tick to start
        await asyncio.sleep(0.01)

        # Stop — signals shutdown, start() drains active tasks
        await worker.stop(timeout=2.0)

        # Wait for start() to complete its cleanup
        await asyncio.wait_for(worker_task, timeout=2.0)

        # Active task should have completed and been cleaned up
        assert completed.is_set()
        assert worker.active_count == 0


class TestExecutePendingRun:
    """Tests for the execute_pending_run() public execution seam."""

    async def test_executes_pending_run(self) -> None:
        """execute_pending_run claims, executes, and finalizes."""
        from dendrite.agent import Agent
        from dendrite.llm import MockLLM
        from dendrite.runtime.runner import execute_pending_run
        from dendrite.types import LLMResponse

        store = WorkerMockStore()
        store.runs["r1"] = _RunRecord(
            id="r1",
            agent_name="Agent",
            status="pending",
            input_data={"input": "hello"},
        )

        agent = Agent(model="mock", prompt="Test.", tools=[])
        llm = MockLLM([LLMResponse(text="world")])

        result = await execute_pending_run(
            "r1",
            state_store=store,  # type: ignore[arg-type]
            agent=agent,
            provider=llm,
        )

        assert result.answer == "world"
        assert result.status.value == "success"
        assert "r1" in store.claimed

    async def test_rejects_nonexistent_run(self) -> None:
        from dendrite.agent import Agent
        from dendrite.llm import MockLLM
        from dendrite.runtime.runner import execute_pending_run
        from dendrite.types import LLMResponse

        store = WorkerMockStore()
        agent = Agent(model="mock", prompt="Test.", tools=[])
        llm = MockLLM([LLMResponse(text="x")])

        with pytest.raises(ValueError, match="not found"):
            await execute_pending_run(
                "nonexistent",
                state_store=store,  # type: ignore[arg-type]
                agent=agent,
                provider=llm,
            )

    async def test_rejects_non_pending_run(self) -> None:
        from dendrite.agent import Agent
        from dendrite.llm import MockLLM
        from dendrite.runtime.runner import execute_pending_run
        from dendrite.types import LLMResponse

        store = WorkerMockStore()
        store.runs["r1"] = _RunRecord(
            id="r1",
            agent_name="Agent",
            status="running",
            input_data={"input": "hello"},
        )

        agent = Agent(model="mock", prompt="Test.", tools=[])
        llm = MockLLM([LLMResponse(text="x")])

        with pytest.raises(ValueError, match="only pending"):
            await execute_pending_run(
                "r1",
                state_store=store,  # type: ignore[arg-type]
                agent=agent,
                provider=llm,
            )

    async def test_rejects_missing_input_data(self) -> None:
        from dendrite.agent import Agent
        from dendrite.llm import MockLLM
        from dendrite.runtime.runner import execute_pending_run
        from dendrite.types import LLMResponse

        store = WorkerMockStore()
        store.runs["r1"] = _RunRecord(
            id="r1",
            agent_name="Agent",
            status="pending",
            input_data=None,
        )

        agent = Agent(model="mock", prompt="Test.", tools=[])
        llm = MockLLM([LLMResponse(text="x")])

        with pytest.raises(ValueError, match="no input_data"):
            await execute_pending_run(
                "r1",
                state_store=store,  # type: ignore[arg-type]
                agent=agent,
                provider=llm,
            )

    async def test_rejects_missing_input_key(self) -> None:
        from dendrite.agent import Agent
        from dendrite.llm import MockLLM
        from dendrite.runtime.runner import execute_pending_run
        from dendrite.types import LLMResponse

        store = WorkerMockStore()
        store.runs["r1"] = _RunRecord(
            id="r1",
            agent_name="Agent",
            status="pending",
            input_data={"other_key": "value"},
        )

        agent = Agent(model="mock", prompt="Test.", tools=[])
        llm = MockLLM([LLMResponse(text="x")])

        with pytest.raises(ValueError, match="no input_data"):
            await execute_pending_run(
                "r1",
                state_store=store,  # type: ignore[arg-type]
                agent=agent,
                provider=llm,
            )
