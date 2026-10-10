# SPDX-FileCopyrightText: Copyright (c) 2026 The DiscoDemo Authors
# SPDX-License-Identifier: Apache-2.0

"""Workcell parameters loaded from ``assets/workcell.json``.

Every physical value is a knob ``{"value": ..., "source": ...}``; bare numbers are
rejected at load time. Sources:

- ``placeholder``: no evidence, a plausible default;
- ``reference``: from literature or vendor data, not measured on this cell;
- ``measured``: measured on this cell or identified from its data.
"""

import json
import os
from typing import Any, Literal, TypedDict

WORKCELL_JSON = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "workcell.json")

_VALID_SOURCES = ("placeholder", "reference", "measured")
# Top-level note without physical values.
_META_KEYS = ("_note",)
# Structural keys (wiring and identifiers) that need no source.
_STRUCTURAL_KEYS = (
    "model",
    "gripper",
    "mount",
    "prim",
    "hw",
    "serial",
    "flip_180",
    "type",
    "dict",
    "id",
)


class Knob(TypedDict):
    """One knob: a value and its source."""

    value: Any
    source: Literal["placeholder", "reference", "measured"]


def _is_knob(node: Any) -> bool:
    return isinstance(node, dict) and "value" in node and "source" in node


def _validate(node: Any, path: str) -> None:
    """Recursively check knob sources and reject bare numeric leaves."""
    if _is_knob(node):
        if node["source"] not in _VALID_SOURCES:
            raise ValueError(f"{path}: invalid source {node['source']!r} (allowed: {_VALID_SOURCES})")
        return
    if isinstance(node, dict):
        for key, child in node.items():
            if key in _STRUCTURAL_KEYS:
                continue
            _validate(child, f"{path}.{key}" if path else key)
        return
    if isinstance(node, (int, float)) and not isinstance(node, bool):
        raise ValueError(f"{path}: physical value without a source; wrap it as {{value, source}}")


def load_workcell(path: str = WORKCELL_JSON) -> dict[str, Any]:
    """Load a workcell json and check that every knob has a valid source."""
    with open(path) as fh:
        doc: dict[str, Any] = json.load(fh)
    for key, node in doc.items():
        if key in _META_KEYS:
            continue
        _validate(node, key)
    return doc


def knob(doc: dict[str, Any], dotted: str) -> Any:
    """Return the value at a dotted path (the node itself if it is not a knob)."""
    node: Any = doc
    for part in dotted.split("."):
        node = node[part]
    return node["value"] if _is_knob(node) else node
