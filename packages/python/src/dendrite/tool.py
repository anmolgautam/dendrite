"""Tool decorator and schema generation.

The @tool decorator turns a plain Python function into a Dendrite tool
by attaching a ToolDef to it. The function itself is unchanged — it stays
callable as normal. Schema is auto-generated from type hints.
"""

from __future__ import annotations

import inspect
import typing
import warnings
from typing import TYPE_CHECKING, Any, get_args, get_origin, get_type_hints

if TYPE_CHECKING:
    from collections.abc import Callable

from dendrite.types import ToolDef, ToolTarget

_TOOL_DEF_ATTR = "__tool_def__"

_TYPE_MAP: dict[type, str] = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
    dict: "object",
    list: "array",
}


def tool(
    target: ToolTarget | str = ToolTarget.SERVER,
    parallel: bool = True,
    priority: int = 0,
    max_calls_per_run: int | None = None,
    timeout_seconds: float = 30.0,
) -> Callable[..., Any]:
    """Decorator that registers a function as a Dendrite tool.

    Usage:
        @tool(target="server")
        async def add(a: int, b: int) -> int:
            '''Add two numbers.'''
            return a + b
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        schema = _generate_schema(fn)
        tool_def = ToolDef(
            name=fn.__name__,
            description=(fn.__doc__ or "").strip(),
            parameters=schema,
            target=ToolTarget(target),
            parallel=parallel,
            priority=priority,
            max_calls_per_run=max_calls_per_run,
            timeout_seconds=timeout_seconds,
        )
        setattr(fn, _TOOL_DEF_ATTR, tool_def)
        return fn

    return decorator


def get_tool_def(fn: Callable[..., Any]) -> ToolDef:
    """Get ToolDef from a decorated function. Raises ValueError if not a tool."""
    tool_def: ToolDef | None = getattr(fn, _TOOL_DEF_ATTR, None)
    if tool_def is None:
        raise ValueError(f"'{fn.__name__}' is not a Dendrite tool. Decorate it with @tool().")
    return tool_def


def is_tool(fn: Callable[..., Any]) -> bool:
    """Check if a function is decorated with @tool."""
    return hasattr(fn, _TOOL_DEF_ATTR)


def _generate_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    """Generate JSON Schema from a function's type hints."""
    sig = inspect.signature(fn)
    hints = get_type_hints(fn)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name == "self":
            continue

        hint = hints.get(name)
        if hint is None:
            raise TypeError(
                f"Parameter '{name}' of tool '{fn.__name__}' has no type hint. "
                f"All tool parameters must be typed for schema generation."
            )

        prop = _type_to_schema(hint)

        has_default = param.default is not inspect.Parameter.empty
        if has_default:
            prop["default"] = param.default
        elif not _is_optional(hint):
            required.append(name)

        properties[name] = prop

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _type_to_schema(hint: type) -> dict[str, Any]:
    """Convert a Python type hint to a JSON Schema property."""
    if _is_optional(hint):
        inner = _unwrap_optional(hint)
        return _type_to_schema(inner)

    origin = get_origin(hint)

    if origin is list:
        args = get_args(hint)
        schema: dict[str, Any] = {"type": "array"}
        if args:
            schema["items"] = _type_to_schema(args[0])
        return schema

    if origin is dict:
        return {"type": "object"}

    json_type = _TYPE_MAP.get(hint)
    if json_type:
        return {"type": json_type}

    warnings.warn(
        f"Type {hint!r} is not recognized by Dendrite schema generation; "
        f"defaulting to {{'type': 'string'}}.",
        stacklevel=2,
    )
    return {"type": "string"}


def _is_optional(hint: type) -> bool:
    """Check if a type hint is Optional (i.e., X | None or Optional[X])."""
    origin = get_origin(hint)
    if origin is not type(int | str) and origin is not typing.Union:
        return False
    args = get_args(hint)
    return type(None) in args


def _unwrap_optional(hint: type) -> type:
    """Extract the inner type from Optional[X] / X | None.

    Note: For multi-type unions like ``str | int | None``, only the first
    non-None type is used. Full JSON Schema ``oneOf`` support is deferred.
    """
    args = get_args(hint)
    non_none = [a for a in args if a is not type(None)]
    return non_none[0] if non_none else str
