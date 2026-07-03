from __future__ import annotations

import json
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult, ContentBlock


class MCPToolError(RuntimeError):
    """Raised when an MCP server reports a tool execution error."""


async def call_tool(server_url: str, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Call a tool on a remote MCP server over streamable HTTP and return its result as a dict.

    Opens a fresh session per call — these tools run once per specimen exception, not in a
    hot loop, so a short-lived connection is simpler than managing a pooled session lifecycle.
    """
    async with streamablehttp_client(server_url) as (read, write, _get_session_id):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)

    return _unwrap(tool_name, result)


def _unwrap(tool_name: str, result: CallToolResult) -> dict[str, Any]:
    if result.isError:
        raise MCPToolError(f"{tool_name} failed: {_text(result.content)}")

    if result.structuredContent is not None:
        return result.structuredContent

    text = _text(result.content)
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"result": text}


def _text(blocks: list[ContentBlock]) -> str:
    return "; ".join(getattr(block, "text", "") for block in blocks if getattr(block, "text", ""))
