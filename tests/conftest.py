from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
os.environ.setdefault("CHECKPOINT_DATABASE_URL", "postgresql://test:test@localhost:5432/test")
os.environ.setdefault("LIMS_MCP_SERVER_URL", "http://localhost:9101/mcp")
os.environ.setdefault("EHR_MCP_SERVER_URL", "http://localhost:9102/mcp")
