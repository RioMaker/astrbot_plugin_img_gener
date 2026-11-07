from __future__ import annotations

import asyncio
import importlib
import sys
from types import SimpleNamespace

import pytest
from test_main_background import _install_astrbot_stubs


@pytest.fixture
def reference_plugin(tmp_path):
    _install_astrbot_stubs()
    sys.modules.pop("astrbot_plugin_img_gener.main", None)
    module = importlib.import_module("astrbot_plugin_img_gener.main")
    plugin = object.__new__(module.GPTImageGeneratorPlugin)
    plugin.config = {}
    plugin.references = module.CharacterReferenceManager(
        [], data_dir=tmp_path, plugin_dir=tmp_path, max_total_images=4
    )
    return module, plugin


@pytest.mark.parametrize("attached_enabled", [True, False])
@pytest.mark.parametrize("quoted_enabled", [True, False])
def test_message_reference_switches_are_independent(
    reference_plugin, tmp_path, attached_enabled, quoted_enabled
):
    module, plugin = reference_plugin
    plugin.config = {
        "references": {
            "enable_attached_images": attached_enabled,
            "enable_quoted_images": quoted_enabled,
        }
    }
    attached = tmp_path / "attached.png"
    quoted = tmp_path / "quoted.png"
    attached_bytes = b"\x89PNG\r\n\x1a\nattached"
    quoted_bytes = b"\x89PNG\r\n\x1a\nquoted"
    if attached_enabled:
        attached.write_bytes(attached_bytes)
    if quoted_enabled:
        quoted.write_bytes(quoted_bytes)
    attached_component = module.Comp.Image(str(attached))
    quoted_component = module.Comp.Image(str(quoted))
    event = SimpleNamespace(
        get_messages=lambda: [
            module.Comp.Reply(chain=[quoted_component]),
            attached_component,
        ]
    )
    sources = asyncio.run(plugin._capture_message_reference_sources(event))
    expected = []
    if attached_enabled:
        expected.append(("attached_1.png", attached_bytes, "image/png"))
    if quoted_enabled:
        expected.append(("quoted_1.png", quoted_bytes, "image/png"))
    assert list(sources) == expected
    # Manual portrait replacement still reads both sources independently of generation switches.
    assert plugin._event_image_components(event) == (quoted_component, attached_component)


@pytest.mark.parametrize(
    "attached_count,quoted_count,expected_error",
    [
        (3, 1, None),
        (4, 1, "超过单次参考图上限 4 张"),
        (1, 2, "单次最多使用 1 张引用消息图片"),
    ],
)
def test_attachment_and_reply_limits_share_total_but_keep_quote_limit(
    reference_plugin, tmp_path, attached_count, quoted_count, expected_error
):
    module, plugin = reference_plugin
    plugin.config = {"references": {"max_quoted_images": 1}}
    path = tmp_path / "reference.png"
    # Over-limit requests must be rejected before any image download/read.
    if not expected_error:
        path.write_bytes(b"\x89PNG\r\n\x1a\nreference")
    event = SimpleNamespace(
        get_messages=lambda: [
            *[module.Comp.Image(str(path)) for _ in range(attached_count)],
            module.Comp.Reply(chain=[module.Comp.Image(str(path)) for _ in range(quoted_count)]),
        ]
    )
    if expected_error:
        with pytest.raises(module.ImageGeneratorError, match=expected_error):
            asyncio.run(plugin._capture_message_reference_sources(event))
    else:
        sources = asyncio.run(plugin._capture_message_reference_sources(event))
        assert len(sources) == 4
        assert [name for name, _, _ in sources] == [
            "attached_1.png",
            "attached_2.png",
            "attached_3.png",
            "quoted_1.png",
        ]


def test_oversized_attachment_is_rejected(reference_plugin, tmp_path):
    module, plugin = reference_plugin
    plugin.config = {"references": {"max_image_size_mb": 1}}
    path = tmp_path / "oversized.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * (1024 * 1024))
    event = SimpleNamespace(get_messages=lambda: [module.Comp.Image(str(path))])
    with pytest.raises(module.ImageGeneratorError, match="当前消息附带的图片超过单张大小上限"):
        asyncio.run(plugin._capture_message_reference_sources(event))


def test_message_images_cannot_silently_displace_all_character_references(
    reference_plugin, tmp_path
):
    module, plugin = reference_plugin
    path = tmp_path / "character.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\ncharacter")
    manager = module.CharacterReferenceManager(
        [{"name": "可可子", "reference_images": [str(path)]}],
        data_dir=tmp_path,
        plugin_dir=tmp_path,
    )
    # All available slots have already been reserved for message images.
    resolved = manager.resolve("可可子挥手", max_total_images=0)
    assert not resolved.image_paths
    assert resolved.capped_characters == ("可可子",)
    error = plugin._reference_resolution_error(resolved, portrait_user_qqs=())
    assert "可可子" in error and "超过了单次参考图上限" in error
    plugin.config = {"references": {"strict_references": False}}
    assert plugin._reference_resolution_error(resolved, portrait_user_qqs=()) is None
