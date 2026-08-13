from __future__ import annotations

from astrbot_plugin_img_gener.config import normalize_base_url


def test_uuapi_cc_is_the_default_base_url() -> None:
    assert normalize_base_url(None) == "https://uuapi.cc/v1"


def test_explicit_uuapi_net_url_is_not_rewritten() -> None:
    assert normalize_base_url("https://uuapi.net/v1") == "https://uuapi.net/v1"
    assert normalize_base_url("https://uuapi.net/") == "https://uuapi.net/v1"


def test_custom_base_url_is_preserved() -> None:
    assert normalize_base_url("https://images.example.com") == (
        "https://images.example.com/v1"
    )
