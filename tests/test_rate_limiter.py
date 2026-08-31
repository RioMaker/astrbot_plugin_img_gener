from __future__ import annotations

import asyncio

from astrbot_plugin_img_gener.rate_limiter import LimitConfig, PersistentRateLimiter


def test_user_cooldown_counts_attempts(tmp_path) -> None:
    limiter = PersistentRateLimiter(
        tmp_path / "limits.sqlite3",
        LimitConfig(
            user_cooldown_seconds=60,
            user_max_attempts=10,
            global_max_concurrent=5,
        ),
    )

    async def scenario() -> None:
        first = await limiter.acquire("qq:1", "qq:100", now=100)
        assert first.allowed
        await limiter.cancel(first.lease_id)
        second = await limiter.acquire("qq:1", "qq:100", now=120)
        assert second.allowed is False
        assert second.code == "USER_COOLDOWN"
        assert second.retry_after == 40

    asyncio.run(scenario())


def test_generation_quota_is_only_committed_on_success(tmp_path) -> None:
    limiter = PersistentRateLimiter(
        tmp_path / "limits.sqlite3",
        LimitConfig(
            user_cooldown_seconds=0,
            group_cooldown_seconds=0,
            user_max_attempts=0,
            user_max_generations=1,
            group_max_generations=10,
            global_max_concurrent=5,
        ),
    )

    async def scenario() -> None:
        failed = await limiter.acquire("qq:1", "qq:100", now=100)
        assert failed.allowed
        await limiter.cancel(failed.lease_id)

        successful = await limiter.acquire("qq:1", "qq:100", now=101)
        assert successful.allowed
        await limiter.complete(successful.lease_id, now=102)

        exhausted = await limiter.acquire("qq:1", "qq:100", now=103)
        assert exhausted.allowed is False
        assert exhausted.code == "USER_GENERATION_LIMIT"

    asyncio.run(scenario())


def test_reservation_prevents_concurrent_group_overshoot(tmp_path) -> None:
    limiter = PersistentRateLimiter(
        tmp_path / "limits.sqlite3",
        LimitConfig(
            user_cooldown_seconds=0,
            group_cooldown_seconds=0,
            user_max_attempts=0,
            user_max_generations=10,
            group_max_generations=10,
            global_max_concurrent=5,
            group_max_concurrent=1,
        ),
    )

    async def scenario() -> None:
        first = await limiter.acquire("qq:1", "qq:100", now=100)
        assert first.allowed
        second = await limiter.acquire("qq:2", "qq:100", now=100)
        assert second.allowed is False
        assert second.code == "GROUP_BUSY"
        await limiter.cancel(first.lease_id)

    asyncio.run(scenario())


def test_limits_persist_across_instances(tmp_path) -> None:
    path = tmp_path / "limits.sqlite3"
    limits = LimitConfig(
        user_cooldown_seconds=0,
        group_cooldown_seconds=0,
        user_max_attempts=0,
        user_max_generations=1,
        group_max_generations=10,
    )

    async def scenario() -> None:
        first_instance = PersistentRateLimiter(path, limits)
        lease = await first_instance.acquire("qq:1", None, now=100)
        await first_instance.complete(lease.lease_id, now=101)

        second_instance = PersistentRateLimiter(path, limits)
        result = await second_instance.acquire("qq:1", None, now=102)
        assert result.allowed is False
        assert result.code == "USER_GENERATION_LIMIT"

    asyncio.run(scenario())

def test_status_reports_remaining_cooldowns_and_pending(tmp_path) -> None:
    limiter = PersistentRateLimiter(
        tmp_path / "limits.sqlite3",
        LimitConfig(
            user_cooldown_seconds=60,
            group_cooldown_seconds=30,
            user_attempt_window_seconds=600,
            user_max_attempts=2,
            user_generation_window_seconds=300,
            user_max_generations=3,
            group_generation_window_seconds=300,
            group_max_generations=5,
            global_max_concurrent=5,
            group_max_concurrent=5,
        ),
    )

    async def scenario() -> None:
        lease = await limiter.acquire("qq:1", "qq:100", now=100)
        status = await limiter.status("qq:1", "qq:100", now=120)

        assert status["user_attempts"] == 1
        assert status["user_attempts_remaining"] == 1
        assert status["user_cooldown_remaining"] == 40
        assert status["group_cooldown_remaining"] == 10
        assert status["user_pending"] == 1
        assert status["user_generations_remaining"] == 2
        assert status["group_pending"] == 1
        assert status["group_generations_remaining"] == 4
        assert status["global_inflight"] == 1

        await limiter.complete(lease.lease_id, now=130)
        completed = await limiter.status("qq:1", "qq:100", now=140)
        assert completed["user_generations"] == 1
        assert completed["user_pending"] == 0
        assert completed["user_generations_remaining"] == 2

    asyncio.run(scenario())


def test_status_reports_quota_window_reset_time(tmp_path) -> None:
    limiter = PersistentRateLimiter(
        tmp_path / "limits.sqlite3",
        LimitConfig(
            user_cooldown_seconds=0,
            group_cooldown_seconds=0,
            user_max_attempts=0,
            user_generation_window_seconds=300,
            user_max_generations=1,
            group_generation_window_seconds=600,
            group_max_generations=1,
        ),
    )

    async def scenario() -> None:
        lease = await limiter.acquire("qq:1", "qq:100", now=100)
        await limiter.complete(lease.lease_id, now=110)

        status = await limiter.status("qq:1", "qq:100", now=120)

        assert status["user_generations_remaining"] == 0
        assert status["user_generation_reset_after"] == 290
        assert status["group_generations_remaining"] == 0
        assert status["group_generation_reset_after"] == 590

    asyncio.run(scenario())
