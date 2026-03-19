"""Tests for ExecutionLease — the shared execution ownership abstraction."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from dendrite.runtime.lease import ExecutionLease

# ------------------------------------------------------------------
# Mock StateStore for lease tests
# ------------------------------------------------------------------


@dataclass
class MockLeaseStore:
    """Minimal mock implementing only the lease-related StateStore methods."""

    _claimed: dict[str, str] = field(default_factory=dict)  # run_id → nonce
    _heartbeats: list[tuple[str, str]] = field(default_factory=list)
    _released: list[tuple[str, str]] = field(default_factory=list)
    claim_should_fail: bool = False
    heartbeat_should_fail: bool = False

    async def claim_run(self, run_id: str, executor_id: str) -> str | None:
        if self.claim_should_fail or run_id in self._claimed:
            return None
        nonce = f"nonce-{len(self._claimed)}"
        self._claimed[run_id] = nonce
        return nonce

    async def renew_heartbeat(self, run_id: str, lease_nonce: str) -> bool:
        if self.heartbeat_should_fail:
            return False
        if self._claimed.get(run_id) != lease_nonce:
            return False
        self._heartbeats.append((run_id, lease_nonce))
        return True

    async def release_lease(self, run_id: str, lease_nonce: str) -> None:
        self._released.append((run_id, lease_nonce))
        if self._claimed.get(run_id) == lease_nonce:
            del self._claimed[run_id]


# ------------------------------------------------------------------
# Acquire
# ------------------------------------------------------------------


class TestAcquire:
    async def test_acquire_returns_lease(self) -> None:
        store = MockLeaseStore()
        lease = await ExecutionLease.acquire(store, "run_1", "exec-A")  # type: ignore[arg-type]
        assert lease is not None
        assert lease.run_id == "run_1"
        assert lease.nonce == "nonce-0"
        assert lease.is_valid is True

    async def test_acquire_fails_returns_none(self) -> None:
        store = MockLeaseStore(claim_should_fail=True)
        lease = await ExecutionLease.acquire(store, "run_1", "exec-A")  # type: ignore[arg-type]
        assert lease is None

    async def test_double_acquire_fails(self) -> None:
        store = MockLeaseStore()
        lease1 = await ExecutionLease.acquire(store, "run_1", "exec-A")  # type: ignore[arg-type]
        assert lease1 is not None
        lease2 = await ExecutionLease.acquire(store, "run_1", "exec-B")  # type: ignore[arg-type]
        assert lease2 is None


# ------------------------------------------------------------------
# Heartbeat
# ------------------------------------------------------------------


class TestHeartbeat:
    async def test_heartbeat_runs_in_background(self) -> None:
        store = MockLeaseStore()
        lease = await ExecutionLease.acquire(
            store,
            "run_1",
            "exec-A",
            heartbeat_interval=0,  # type: ignore[arg-type]
        )
        assert lease is not None

        lease.start_heartbeat()
        await asyncio.sleep(0.05)  # let a few heartbeats fire
        await lease.stop_heartbeat()

        assert len(store._heartbeats) > 0
        assert all(r == "run_1" for r, _ in store._heartbeats)

    async def test_heartbeat_detects_superseded_nonce(self) -> None:
        store = MockLeaseStore()
        lease = await ExecutionLease.acquire(
            store,
            "run_1",
            "exec-A",
            heartbeat_interval=0,  # type: ignore[arg-type]
        )
        assert lease is not None

        # Simulate another executor reclaiming
        store.heartbeat_should_fail = True

        lease.start_heartbeat()
        await asyncio.sleep(0.05)

        assert lease.is_valid is False
        await lease.stop_heartbeat()

    async def test_start_heartbeat_idempotent(self) -> None:
        store = MockLeaseStore()
        lease = await ExecutionLease.acquire(store, "run_1", "exec-A")  # type: ignore[arg-type]
        assert lease is not None

        lease.start_heartbeat()
        lease.start_heartbeat()  # second call should be no-op
        await lease.stop_heartbeat()

    async def test_stop_without_start_is_noop(self) -> None:
        store = MockLeaseStore()
        lease = await ExecutionLease.acquire(store, "run_1", "exec-A")  # type: ignore[arg-type]
        assert lease is not None
        await lease.stop_heartbeat()  # should not raise


# ------------------------------------------------------------------
# Release
# ------------------------------------------------------------------


class TestRelease:
    async def test_release_clears_lease(self) -> None:
        store = MockLeaseStore()
        lease = await ExecutionLease.acquire(store, "run_1", "exec-A")  # type: ignore[arg-type]
        assert lease is not None

        await lease.release()
        assert ("run_1", "nonce-0") in store._released
        assert "run_1" not in store._claimed

    async def test_release_stops_heartbeat(self) -> None:
        store = MockLeaseStore()
        lease = await ExecutionLease.acquire(
            store,
            "run_1",
            "exec-A",
            heartbeat_interval=0,  # type: ignore[arg-type]
        )
        assert lease is not None

        lease.start_heartbeat()
        await lease.release()

        # Heartbeat should be stopped — no more writes
        count_before = len(store._heartbeats)
        await asyncio.sleep(0.05)
        assert len(store._heartbeats) == count_before


# ------------------------------------------------------------------
# Nonce property
# ------------------------------------------------------------------


class TestNonceProperty:
    async def test_nonce_is_stable(self) -> None:
        store = MockLeaseStore()
        lease = await ExecutionLease.acquire(store, "run_1", "exec-A")  # type: ignore[arg-type]
        assert lease is not None
        assert lease.nonce == lease.nonce  # same value on repeated access
