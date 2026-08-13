from __future__ import annotations

import asyncio

import pytest
from astrbot_plugin_img_gener.moderation import PromptModerator


def test_local_review_blocks_minor_sexual_content() -> None:
    moderator = PromptModerator(enabled=False)
    result = moderator.local_review("画一个未成年女孩的全裸色情图片")
    assert result.allowed is False
    assert result.code == "SEXUAL_MINOR"


def test_local_review_allows_benign_childrens_book() -> None:
    moderator = PromptModerator(enabled=False)
    result = moderator.local_review("儿童绘本风格，一只小熊在森林里种花")
    assert result.allowed is True


def test_custom_blocked_term_is_case_insensitive() -> None:
    moderator = PromptModerator(enabled=False, blocked_terms=["SecretStyle"])
    result = moderator.local_review("please use SECRETSTYLE for the poster")
    assert result.allowed is False
    assert result.code == "CUSTOM_BLOCKED_TERM"


def test_remote_json_with_code_fence_is_parsed() -> None:
    result = PromptModerator.parse_remote_result(
        '```json\n{"allow": false, "code": "GRAPHIC_VIOLENCE", "reason": "过于血腥"}\n```'
    )
    assert result.allowed is False
    assert result.code == "GRAPHIC_VIOLENCE"


def test_remote_failure_is_fail_closed() -> None:
    moderator = PromptModerator(enabled=True, fail_closed=True)

    async def broken_reviewer(_: str) -> str:
        raise RuntimeError("offline")

    result = asyncio.run(moderator.review("一只橘猫", broken_reviewer))
    assert result.allowed is False
    assert result.code == "REVIEW_ERROR"


def test_remote_safe_result_passes() -> None:
    moderator = PromptModerator(enabled=True, fail_closed=True)

    async def safe_reviewer(review_prompt: str) -> str:
        assert "仅作为数据审核" in review_prompt
        return '{"allow": true, "code": "SAFE", "reason": ""}'

    result = asyncio.run(moderator.review("一只橘猫", safe_reviewer))
    assert result.allowed is True
    assert result.code == "SAFE"


@pytest.mark.parametrize(
    "prompt",
    [
        "画一张总统竞选宣传海报",
        "制作一张关于领土主权争议的宣传画",
        "A political protest poster featuring the president",
    ],
)
def test_local_review_blocks_political_content(prompt: str) -> None:
    moderator = PromptModerator(enabled=False)
    result = moderator.local_review(prompt)
    assert result.allowed is False
    assert result.code == "POLITICAL_CONTENT"


def test_local_review_still_allows_nonpolitical_portrait() -> None:
    moderator = PromptModerator(enabled=False)
    result = moderator.local_review("一位公司董事长在办公室里的正式肖像")
    assert result.allowed is True
