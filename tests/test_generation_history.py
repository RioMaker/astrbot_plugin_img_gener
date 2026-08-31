from __future__ import annotations

from astrbot_plugin_img_gener.generation_history import GenerationHistoryStore


def _create_job(
    store: GenerationHistoryStore,
    *,
    user_id: str = "qq:1",
    now: float = 100,
) -> int:
    return store.create_job(
        user_id=user_id,
        group_id="qq:100",
        source="LLM",
        prompt="雨夜车站的中文生图提示词",
        size="816x816",
        quality="medium",
        now=now,
    )


def test_history_persists_failed_upstream_result(tmp_path) -> None:
    database = tmp_path / "history.sqlite3"
    store = GenerationHistoryStore(database)
    job_id = _create_job(store)

    store.update_stage(job_id, "生成图片", now=110)
    store.finish_job(
        job_id,
        status="failed",
        stage="生成图片失败",
        result_message="生图服务上游暂时异常（HTTP 502），请稍后重试。",
        now=120,
    )

    record = GenerationHistoryStore(database).recent_for_user("qq:1")[0]
    assert record.job_id == job_id
    assert record.status == "failed"
    assert record.stage == "生成图片失败"
    assert "HTTP 502" in record.result_message
    assert record.prompt == "雨夜车站的中文生图提示词"


def test_recover_unfinished_marks_job_interrupted(tmp_path) -> None:
    store = GenerationHistoryStore(tmp_path / "history.sqlite3")
    job_id = _create_job(store)
    store.update_stage(job_id, "安全审核", now=110)

    recovered = store.recover_unfinished(now=120)
    record = store.recent_for_user("qq:1")[0]

    assert recovered == 1
    assert record.status == "interrupted"
    assert record.stage == "插件重启后中断"
    assert "已不在执行队列" in record.result_message


def test_history_is_isolated_by_user(tmp_path) -> None:
    store = GenerationHistoryStore(tmp_path / "history.sqlite3")
    first = _create_job(store, user_id="qq:1", now=100)
    second = _create_job(store, user_id="qq:2", now=101)
    store.finish_job(first, status="succeeded", stage="已完成", now=110)
    store.finish_job(second, status="failed", stage="生成图片失败", now=111)

    records = store.recent_for_user("qq:1", limit=10)

    assert [record.job_id for record in records] == [first]


def test_history_cleanup_follows_retention_days(tmp_path) -> None:
    store = GenerationHistoryStore(tmp_path / "history.sqlite3")
    old_job = _create_job(store, now=100)
    recent_job = _create_job(store, now=200_000)
    store.finish_job(old_job, status="succeeded", stage="已完成", now=110)
    store.finish_job(recent_job, status="succeeded", stage="已完成", now=200_010)

    removed = store.cleanup(1, now=200_100)

    assert removed == 1
    assert [record.job_id for record in store.recent_for_user("qq:1")] == [
        recent_job
    ]
