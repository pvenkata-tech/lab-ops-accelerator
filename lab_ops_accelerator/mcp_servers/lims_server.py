from __future__ import annotations

import logging
import os

import httpx
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LIMS_API_BASE_URL = os.environ.get("LIMS_API_BASE_URL", "http://lims-mock:8001")
LIMS_API_KEY = os.environ.get("LIMS_API_KEY", "")

mcp = FastMCP(
    "lims-server",
    host=os.environ.get("MCP_HOST", "0.0.0.0"),
    port=int(os.environ.get("LIMS_MCP_SERVER_PORT", "9101")),
)


@mcp.tool()
async def update_specimen_disposition(
    specimen_id: str,
    order_id: str,
    disposition: str,
    requires_retest: bool,
    protocol_applied: str | None = None,
    confidence: float | None = None,
) -> dict:
    """Update a specimen's disposition and status in the LIMS after exception resolution."""
    payload = {
        "specimen_id": specimen_id,
        "order_id": order_id,
        "disposition": disposition,
        "protocol_applied": protocol_applied,
        "confidence": confidence,
        "requires_retest": requires_retest,
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            f"{LIMS_API_BASE_URL}/specimens/{specimen_id}/disposition",
            json=payload,
            headers={"X-API-Key": LIMS_API_KEY},
        )
        response.raise_for_status()
    logger.info("Updated LIMS disposition for specimen %s: %s", specimen_id, disposition)
    return {"status": "updated", "specimen_id": specimen_id}


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
