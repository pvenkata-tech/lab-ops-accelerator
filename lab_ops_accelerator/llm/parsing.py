from __future__ import annotations

import re

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def strip_code_fence(text: str) -> str:
    """Strip a leading/trailing markdown code fence some LLMs wrap JSON responses in.

    e.g. turns '```json\\n{"a": 1}\\n```' into '{"a": 1}' so json.loads doesn't choke
    on the fence markers.
    """
    return _CODE_FENCE_RE.sub("", text.strip()).strip()
