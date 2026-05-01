"""Small deterministic JSON bounding helpers for local brain persistence."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

MAX_BRAIN_EVENT_PAYLOAD_JSON_CHARS = 262_144
MAX_CLAIM_OBJECT_JSON_CHARS = 8_192
MAX_BOUNDED_STRING_CHARS = 2_048


def bounded_json_value(
    value: Any,
    *,
    max_depth: int = 6,
    max_items: int = 32,
    max_string_chars: int = MAX_BOUNDED_STRING_CHARS,
) -> Any:
    """Return a deterministic JSON-compatible value with bounded depth and size."""
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return value if len(value) <= max_string_chars else value[:max_string_chars].rstrip()
    if max_depth <= 0:
        return {
            "_payload_status": "bounded_depth",
            "reason_codes": ["json_depth_limit"],
        }
    if isinstance(value, Mapping):
        items = sorted(value.items(), key=lambda item: str(item[0]))
        bounded: dict[str, Any] = {}
        for key, item_value in items[:max_items]:
            bounded[str(key)] = bounded_json_value(
                item_value,
                max_depth=max_depth - 1,
                max_items=max_items,
                max_string_chars=max_string_chars,
            )
        if len(items) > max_items:
            bounded["_payload_status"] = "bounded_items"
            bounded["_omitted_item_count"] = len(items) - max_items
            bounded.setdefault("reason_codes", ["json_item_limit"])
        return bounded
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        sequence = list(value[:max_items])
        bounded_list = [
            bounded_json_value(
                item,
                max_depth=max_depth - 1,
                max_items=max_items,
                max_string_chars=max_string_chars,
            )
            for item in sequence
        ]
        if len(value) > max_items:
            bounded_list.append(
                {
                    "_payload_status": "bounded_items",
                    "_omitted_item_count": len(value) - max_items,
                    "reason_codes": ["json_item_limit"],
                }
            )
        return bounded_list
    return str(value)[:max_string_chars].rstrip()


def dumps_bounded_json(
    value: Any,
    *,
    max_chars: int,
    overflow_kind: str,
    ensure_ascii: bool = False,
    sort_keys: bool = True,
    max_depth: int = 6,
    max_items: int = 32,
    max_string_chars: int = MAX_BOUNDED_STRING_CHARS,
) -> str:
    """Serialize one JSON value after applying deterministic public-safe bounds."""
    bounded = bounded_json_value(
        value,
        max_depth=max_depth,
        max_items=max_items,
        max_string_chars=max_string_chars,
    )
    encoded = json.dumps(bounded, ensure_ascii=ensure_ascii, sort_keys=sort_keys)
    if len(encoded) <= max_chars:
        return encoded
    marker = {
        "_payload_status": "bounded_oversize",
        "bounded_payload_chars": len(encoded),
        "reason_codes": [overflow_kind],
    }
    return json.dumps(marker, ensure_ascii=ensure_ascii, sort_keys=sort_keys)


def safe_json_dict(payload_json: str, *, max_chars: int, overflow_kind: str) -> dict[str, Any]:
    """Decode a JSON object unless it is too large or malformed."""
    raw = payload_json or "{}"
    if len(raw) > max_chars:
        return {
            "_payload_status": "quarantined_oversize",
            "payload_chars": len(raw),
            "reason_codes": [overflow_kind],
        }
    try:
        decoded = json.loads(raw)
    except (TypeError, ValueError):
        return {
            "_payload_status": "invalid_json",
            "reason_codes": ["json_decode_failed"],
        }
    return dict(decoded) if isinstance(decoded, dict) else {}
