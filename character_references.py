from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass(frozen=True, slots=True)
class CharacterReference:
    name: str
    aliases: tuple[str, ...]
    image_paths: tuple[Path, ...]
    prompt_note: str = ""


@dataclass(frozen=True, slots=True)
class ResolvedReferences:
    characters: tuple[CharacterReference, ...]
    image_paths: tuple[Path, ...]
    missing_characters: tuple[str, ...]
    unknown_requested_names: tuple[str, ...]
    capped_characters: tuple[str, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.characters)


class CharacterReferenceManager:
    """Parse WebUI character entries and match names/aliases in a prompt."""

    def __init__(
        self,
        entries: Any,
        *,
        data_dir: Path,
        plugin_dir: Path,
        max_total_images: int = 4,
        max_image_size_mb: int = 12,
        per_character_limit: int = 1,
    ) -> None:
        self.data_dir = data_dir
        self.plugin_dir = plugin_dir
        self.max_total_images = max(1, max_total_images)
        self.max_image_bytes = max(1, max_image_size_mb) * 1024 * 1024
        self.per_character_limit = max(1, per_character_limit)
        self.characters = tuple(self._parse_entries(entries))

    @staticmethod
    def _flatten_file_values(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, dict):
            for key in ("path", "file_path", "local_path", "url", "value"):
                if value.get(key):
                    return CharacterReferenceManager._flatten_file_values(value[key])
            result: list[str] = []
            for item in value.values():
                result.extend(CharacterReferenceManager._flatten_file_values(item))
            return result
        if isinstance(value, (list, tuple, set)):
            result = []
            for item in value:
                result.extend(CharacterReferenceManager._flatten_file_values(item))
            return result
        return []

    def _resolve_path(self, raw: str) -> Path:
        path = Path(raw).expanduser()
        candidates = [path] if path.is_absolute() else [
            self.data_dir / path,
            self.plugin_dir / path,
            Path.cwd() / path,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        return candidates[0]

    def _parse_legacy_entry(self, value: str) -> dict[str, Any] | None:
        if "=" not in value:
            return None
        names, images = value.split("=", 1)
        parts = [part.strip() for part in names.split("|") if part.strip()]
        if not parts:
            return None
        return {
            "enabled": True,
            "name": parts[0],
            "aliases": parts[1:],
            "reference_images": [item.strip() for item in images.split(",")],
            "prompt_note": "",
        }

    def _parse_entries(self, entries: Any) -> list[CharacterReference]:
        if not isinstance(entries, list):
            return []
        parsed: list[CharacterReference] = []
        for raw_entry in entries:
            entry = (
                self._parse_legacy_entry(raw_entry)
                if isinstance(raw_entry, str)
                else raw_entry
            )
            if not isinstance(entry, dict) or entry.get("enabled", True) is False:
                continue
            name = str(entry.get("name") or "").strip()
            if not name:
                continue
            raw_aliases = entry.get("aliases") or []
            if isinstance(raw_aliases, str):
                raw_aliases = raw_aliases.replace("，", ",").split(",")
            aliases = tuple(
                dict.fromkeys(
                    [
                        str(alias).strip()
                        for alias in raw_aliases
                        if str(alias).strip() and str(alias).strip() != name
                    ]
                )
            )
            paths = tuple(
                dict.fromkeys(
                    self._resolve_path(item)
                    for item in self._flatten_file_values(
                        entry.get("reference_images") or entry.get("images")
                    )
                    if item
                )
            )
            parsed.append(
                CharacterReference(
                    name=name,
                    aliases=aliases,
                    image_paths=paths,
                    prompt_note=str(entry.get("prompt_note") or "").strip(),
                )
            )
        return parsed

    @staticmethod
    def _contains_alias(prompt: str, alias: str) -> bool:
        if not alias:
            return False
        if alias.isascii() and re.fullmatch(r"[A-Za-z0-9_\- ]+", alias):
            return bool(
                re.search(rf"(?<![A-Za-z0-9_]){re.escape(alias)}(?![A-Za-z0-9_])", prompt, re.I)
            )
        return alias in prompt

    def _usable_paths(self, character: CharacterReference) -> tuple[Path, ...]:
        result: list[Path] = []
        for path in character.image_paths:
            try:
                if (
                    path.is_file()
                    and path.suffix.lower() in ALLOWED_IMAGE_SUFFIXES
                    and 0 < path.stat().st_size <= self.max_image_bytes
                ):
                    result.append(path)
            except OSError:
                continue
        return tuple(result)

    def resolve(
        self, prompt: str, requested_names: list[str] | None = None
    ) -> ResolvedReferences:
        selected: list[CharacterReference] = []
        unknown: list[str] = []
        explicit = [str(item).strip() for item in requested_names or [] if str(item).strip()]

        if explicit:
            for requested in explicit:
                matched = next(
                    (
                        item
                        for item in self.characters
                        if requested.casefold()
                        in {item.name.casefold(), *(alias.casefold() for alias in item.aliases)}
                    ),
                    None,
                )
                if matched and matched not in selected:
                    selected.append(matched)
                elif matched is None:
                    unknown.append(requested)

        for character in self.characters:
            aliases = sorted(
                (character.name, *character.aliases), key=len, reverse=True
            )
            if any(len(alias) >= 2 and self._contains_alias(prompt, alias) for alias in aliases):  # noqa: SIM102
                if character not in selected:
                    selected.append(character)

        images: list[Path] = []
        used: set[Path] = set()
        missing: list[str] = []
        capped: list[str] = []
        usable_by_character = {
            character: self._usable_paths(character) for character in selected
        }
        multi_character = len(selected) > 1
        per_character_cap = (
            self.per_character_limit if multi_character else self.max_total_images
        )
        next_index: dict[CharacterReference, int] = {}

        # First pass: guarantee at least one image for every selected character.
        # A character that ends up with zero images would otherwise be silently
        # dropped by the model, which is a common cause of "missing character".
        for character in selected:
            usable = usable_by_character[character]
            if not usable:
                missing.append(character.name)
                continue
            if len(images) >= self.max_total_images:
                capped.append(character.name)
                continue
            first = next((path for path in usable if path not in used), usable[0])
            images.append(first)
            used.add(first)
            next_index[character] = 1

        # Second pass: fill the remaining budget round-robin up to the
        # per-character cap, keeping upload order aligned with prompt numbering.
        remaining = self.max_total_images - len(images)
        if remaining > 0:
            active = [character for character in selected if character in next_index]
            while remaining > 0:
                progressed = False
                for character in active:
                    if remaining <= 0:
                        break
                    index = next_index[character]
                    usable = usable_by_character[character]
                    if index >= len(usable) or index >= per_character_cap:
                        continue
                    path = usable[index]
                    if path in used:
                        next_index[character] = index + 1
                        progressed = True
                        continue
                    images.append(path)
                    used.add(path)
                    next_index[character] = index + 1
                    remaining -= 1
                    progressed = True
                if not progressed:
                    break

        return ResolvedReferences(
            characters=tuple(selected),
            image_paths=tuple(images),
            missing_characters=tuple(missing),
            unknown_requested_names=tuple(unknown),
            capped_characters=tuple(capped),
        )

    @staticmethod
    def augment_prompt(
        prompt: str, resolved: ResolvedReferences, *, use_sheet: bool = False
    ) -> str:
        if not resolved.characters:
            return prompt
        lines = [prompt.strip(), "", "角色参考图使用规则："]
        names = "、".join(character.name for character in resolved.characters)
        lines.append(
            f"画面中必须完整出现以下 {len(resolved.characters)} 个角色"
            f"（共 {len(resolved.characters)} 个，缺一不可）：{names}。"
            "不得省略或替换任何角色，也不要把多个角色合并成同一个人。"
        )
        if use_sheet:
            lines.append(
                "上传的是一张拼合参考图，格子按从左到右、从上到下排列；"
                "请按格子顺序对应下列角色。"
            )
        elif len(resolved.characters) == 2:
            lines.append("若用户未指定构图，可让两个角色左右并排、各自独立。")
        image_index = 1
        for character in resolved.characters:
            usable_count = sum(
                1 for path in character.image_paths if path in resolved.image_paths
            )
            if usable_count <= 0:
                if character.prompt_note:
                    lines.append(
                        f"- 角色「{character.name}」未携带参考图，"
                        f"请严格按以下设定绘制：{character.prompt_note}"
                    )
                continue
            end = image_index + usable_count - 1
            if use_sheet:
                if usable_count == 1 and image_index in {1, 2}:
                    position = "（左）" if image_index == 1 else "（右）"
                    image_label = f"第 {image_index} 格{position}"
                else:
                    image_label = (
                        f"第 {image_index} 格"
                        if image_index == end
                        else f"第 {image_index}-{end} 格"
                    )
            else:
                image_label = (
                    f"第 {image_index} 张参考图"
                    if image_index == end
                    else f"第 {image_index}-{end} 张参考图"
                )
            line = (
                f"- {image_label} 对应角色「{character.name}」；"
                "保持其身份、脸部和核心外观一致。"
            )
            if character.prompt_note:
                line += f" 补充设定：{character.prompt_note}"
            lines.append(line)
            image_index = end + 1
        lines.append(
            "参考图仅用于确定每个角色各自的外观；"
            "不要把不同角色的身份特征混合到同一个人身上。"
        )
        return "\n".join(lines)

