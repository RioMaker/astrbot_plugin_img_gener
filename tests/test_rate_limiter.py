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
