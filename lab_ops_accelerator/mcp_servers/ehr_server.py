from __future__ import annotations

import logging
import os

import httpx
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EHR_WEBHOOK_URL = os.environ.get("EHR_WEBHOOK_URL", "http://ehr-mock:8002/notify")
EHR_API_KEY = os.environ.get("EHR_API_KEY", "")

mcp = FastMCP(
    "ehr-server",
    host=os.environ.get("MCP_HOST", "0.0.0.0"),
    port=int(os.environ.get("EHR_MCP_SERVER_PORT", "9102")),
)


@mcp.tool()
async def notify_physician(
    order_id: str,
    patient_id: str,
    disposition: str,
    message: str,
    notification_type: str = "specimen_exception",
) -> dict:
    """Notify the ordering physician's EHR inbox about a specimen exception disposition."""
    payload = {
        "order_id": order_id,
        "patient_id": patient_id,
        "notification_type": notification_type,
        "disposition": disposition,
        "message": message,
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            EHR_WEBHOOK_URL,
            json=payload,
            headers={"X-API-Key": EHR_API_KEY},
        )
        response.raise_for_status()
    logger.info("Notified EHR for order %s: %s", order_id, disposition)
    return {"status": "notified", "order_id": order_id}


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
