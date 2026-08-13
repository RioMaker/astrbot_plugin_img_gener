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
    ) -> None:
        self.data_dir = data_dir
        self.plugin_dir = plugin_dir
        self.max_total_images = max(1, max_total_images)
        self.max_image_bytes = max(1, max_image_size_mb) * 1024 * 1024
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
        missing: list[str] = []
        for character in selected:
            usable = self._usable_paths(character)
            if not usable:
                missing.append(character.name)
                continue
            for path in usable:
                if path not in images and len(images) < self.max_total_images:
                    images.append(path)

        return ResolvedReferences(
            characters=tuple(selected),
            image_paths=tuple(images),
            missing_characters=tuple(missing),
            unknown_requested_names=tuple(unknown),
        )

    @staticmethod
    def augment_prompt(prompt: str, resolved: ResolvedReferences) -> str:
        if not resolved.characters:
            return prompt
        lines = [prompt.strip(), "", "角色参考图使用规则："]
        image_index = 1
        for character in resolved.characters:
            usable_count = sum(
                1 for path in character.image_paths if path in resolved.image_paths
            )
            if usable_count <= 0:
                continue
            end = image_index + usable_count - 1
            image_label = (
                f"参考图 {image_index}" if image_index == end else f"参考图 {image_index}-{end}"
            )
            line = f"- {image_label} 对应角色“{character.name}”；保持其身份、脸部和核心外观一致。"
            if character.prompt_note:
                line += f" 补充设定：{character.prompt_note}"
            lines.append(line)
            image_index = end + 1
        lines.append("除用户明确要求修改的部分外，不要把不同角色的身份特征互相混合。")
        return "\n".join(lines)

