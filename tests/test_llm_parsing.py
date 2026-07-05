from __future__ import annotations

from lab_ops_accelerator.llm.parsing import strip_code_fence


def test_strips_json_code_fence():
    raw = '```json\n{"a": 1}\n```'
    assert strip_code_fence(raw) == '{"a": 1}'


def test_strips_bare_code_fence():
    raw = '```\n{"a": 1}\n```'
    assert strip_code_fence(raw) == '{"a": 1}'


def test_passes_through_plain_json():
    raw = '{"a": 1}'
    assert strip_code_fence(raw) == '{"a": 1}'
