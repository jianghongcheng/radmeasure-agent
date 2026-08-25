"""Small dependency-free MCP stdio server for the GeoMed tool boundary."""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from typing import Any

from .factory import create_tools_from_env
from .tools import GeoMedTools

PROTOCOL_VERSION = "2024-11-05"


@lru_cache(maxsize=1)
def get_tools() -> GeoMedTools:
    return create_tools_from_env()


TOOL_SCHEMAS = [
    {"name": "list_geomed_capabilities", "description": "Describe the GeoMed backend, measurements, and limitations.", "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "list_available_cases", "description": "List case identifiers accepted by the configured backend.", "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20}}, "additionalProperties": False}},
    {
        "name": "analyze_radiograph",
        "description": "Run geometry checks, retrieval, citations, and tool traces for a configured case ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_id": {"type": "string", "minLength": 1},
                "question": {"type": "string", "minLength": 1},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 3},
            },
            "required": ["image_id"],
            "additionalProperties": False,
        },
    },
]


def _result(request_id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def dispatch(message: dict) -> dict | None:
    """Dispatch one MCP JSON-RPC message; notifications return no response."""
    if "id" not in message:
        return None
    request_id = message["id"]
    method = message.get("method")
    params = message.get("params") or {}
    try:
        if method == "initialize":
            return _result(request_id, {"protocolVersion": PROTOCOL_VERSION, "capabilities": {"tools": {"listChanged": False}}, "serverInfo": {"name": "geomed-copilot", "version": "0.3.0"}})
        if method == "ping":
            return _result(request_id, {})
        if method == "tools/list":
            return _result(request_id, {"tools": TOOL_SCHEMAS})
        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments") or {}
            if name == "list_geomed_capabilities":
                output = get_tools().capabilities()
            elif name == "list_available_cases":
                output = get_tools().list_available_cases(**arguments)
            elif name == "analyze_radiograph":
                output = get_tools().analyze_radiograph(**arguments)
            else:
                return _error(request_id, -32602, f"Unknown tool: {name}")
            return _result(request_id, {"content": [{"type": "text", "text": json.dumps(output, ensure_ascii=False)}], "structuredContent": output, "isError": False})
        return _error(request_id, -32601, f"Method not found: {method}")
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        return _result(request_id, {"content": [{"type": "text", "text": str(exc)}], "isError": True})


def main() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            response = dispatch(json.loads(line))
        except (json.JSONDecodeError, TypeError) as exc:
            response = _error(None, -32700, f"Parse error: {exc}")
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":  # pragma: no cover
    main()
