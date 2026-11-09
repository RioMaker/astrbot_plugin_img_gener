"""Offline command contract, with an optional unmodified AstrBot source replay.

ASTRBOT_COMMAND_FILTER_SOURCE may point at AstrBot's core/star/filter/command.py.
The contract models its public GreedyStr/default precedence and wake boundary;
it does not replace the framework's full parser or other filter types.
Source: https://github.com/AstrBotDevs/AstrBot/blob/master/astrbot/core/star/filter/command.py
Verified upstream Git blob: abf339fda7f0740f542a91cd4adb859d5ef6ed5f.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import sys
import types
from pathlib import Path


class GreedyStr(str):
    pass


class CommandFilterContract:
    def __init__(self, command_name, alias=None, handler_md=None):
        self.names = (command_name, *sorted(alias or ()))
        self.handler_md = handler_md
        self.parameters = tuple(
            inspect.signature(handler_md.handler, eval_str=True).parameters.values()
        )[2:]

    def filter(self, event, _config):
        if not event.is_at_or_wake_command:
            return False
        text = " ".join(event.get_message_str().split())
        name = next(
            (name for name in self.names if text == name or text.startswith(name + " ")), None
        )
        if name is None:
            return False
        tokens = text[len(name) :].split()
        parsed = {}
        for index, parameter in enumerate(self.parameters):
            # AstrBot stores a default value instead of the annotation when supplied.
            schema = (
                parameter.annotation
                if parameter.default is inspect.Parameter.empty
                else parameter.default
            )
            if schema is GreedyStr:
                parsed[parameter.name] = " ".join(tokens[index:])
                break
            if index < len(tokens):
                parsed[parameter.name] = tokens[index]
            elif parameter.default is not inspect.Parameter.empty:
                parsed[parameter.name] = parameter.default
            else:
                raise ValueError("Required command argument missing")
        event.set_extra("parsed_params", parsed)
        return True


def install_command_parser(monkeypatch):
    source = os.environ.get("ASTRBOT_COMMAND_FILTER_SOURCE")
    name = "astrbot.core.star.filter.command"
    if not source:
        module = types.ModuleType(name)
        module.CommandFilter = CommandFilterContract
        module.GreedyStr = GreedyStr
        monkeypatch.setitem(sys.modules, name, module)
        return module

    # Stub dependencies only; execute the actual CommandFilter and GreedyStr unchanged.
    dependencies = {
        "astrbot.core.config": {"AstrBotConfig": dict},
        "astrbot.core.platform.astr_message_event": {"AstrMessageEvent": object},
        "astrbot.core.star.star_handler": {"StarHandlerMetadata": types.SimpleNamespace},
        "astrbot.core.star.filter": {"HandlerFilter": object},
        "astrbot.core.star.filter.custom_filter": {"CustomFilter": object},
    }
    for dependency, attributes in dependencies.items():
        module = types.ModuleType(dependency)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, dependency, module)
    spec = importlib.util.spec_from_file_location(name, Path(source).resolve())
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module
