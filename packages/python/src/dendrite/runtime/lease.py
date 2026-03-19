"""ExecutionLease — shared execution ownership abstraction.

Encapsulates claim, heartbeat, and nonce-guarded coordination for one
run. Used by both library mode (run()) and hosted mode (WorkerLoop).
Neither should implement lease logic directly — both go through this.

Lifecycle:
    lease = await ExecutionLease.acquire(store, run_id, executor_id)
    try:
        lease.start_heartbeat()
        # ... execute agent loop, passing lease.nonce to all writes ...
    finally:
        await lease.stop_heartbeat()
        await lease.release()
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dendrite.runtime.state import StateStore

logger = logging.getLogger(__name__)

# Default heartbeat interval. Configurable via constructor.
_DEFAULT_HEARTBEAT_INTERVAL = 10  # seconds


class ExecutionLease:
    """Proof of execution ownership for one run.

    All StateStore writes during execution should pass ``self.nonce``
    as ``lease_nonce`` to ensure stale executors cannot corrupt state.

    The heartbeat runs as a background asyncio.Task, renewing the
    lease every ``heartbeat_interval`` seconds. If the nonce becomes
    stale (another executor reclaimed the run), the heartbeat detects
    it and sets ``self.is_valid`` to False.
    """

    def __init__(
        self,
        state_store: StateStore,
        run_id: str,
        nonce: str,
        *,
        heartbeat_interval: int = _DEFAULT_HEARTBEAT_INTERVAL,
    ) -> None:
        self._store = state_store
        self._run_id = run_id
        self._nonce = nonce
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._valid = True

    @classmethod
    async def acquire(
        cls,
        state_store: StateStore,
        run_id: str,
        executor_id: str,
        *,
        heartbeat_interval: int = _DEFAULT_HEARTBEAT_INTERVAL,
    ) -> ExecutionLease | None:
        """Claim a pending run and return a lease. Returns None if claim fails."""
        nonce = await state_store.claim_run(run_id, executor_id)
        if nonce is None:
            return None
        return cls(
            state_store,
            run_id,
            nonce,
            heartbeat_interval=heartbeat_interval,
        )

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def nonce(self) -> str:
        """The lease nonce. Pass this to all guarded StateStore writes."""
        return self._nonce

    @property
    def is_valid(self) -> bool:
        """False if the heartbeat detected a superseded nonce."""
        return self._valid

    def start_heartbeat(self) -> None:
        """Start the background heartbeat task."""
        if self._heartbeat_task is not None:
            return  # already running
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(),
            name=f"heartbeat-{self._run_id}",
        )

    async def stop_heartbeat(self) -> None:
        """Stop the background heartbeat task."""
        if self._heartbeat_task is None:
            return
        self._heartbeat_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._heartbeat_task
        self._heartbeat_task = None

    async def release(self) -> None:
        """Release the lease. Clears executor_id, nonce, heartbeat in DB."""
        await self.stop_heartbeat()
        try:
            await self._store.release_lease(self._run_id, self._nonce)
        except Exception:
            logger.warning("Failed to release lease for run %s", self._run_id, exc_info=True)

    async def _heartbeat_loop(self) -> None:
        """Periodically renew the heartbeat. Stops if nonce is superseded."""
        while True:
            await asyncio.sleep(self._heartbeat_interval)
            try:
                renewed = await self._store.renew_heartbeat(self._run_id, self._nonce)
                if not renewed:
                    logger.warning(
                        "Lease superseded for run %s (nonce %s) — stopping heartbeat",
                        self._run_id,
                        self._nonce[:8],
                    )
                    self._valid = False
                    return
            except Exception:
                logger.warning("Heartbeat failed for run %s", self._run_id, exc_info=True)
