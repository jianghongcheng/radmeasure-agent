import json

from radmeasure import mcp_server
from radmeasure.production import DemoService
from radmeasure.tools import RadMeasureTools


def test_mcp_initialize_list_and_call(monkeypatch):
    monkeypatch.setattr(mcp_server, "get_tools", lambda: RadMeasureTools(DemoService()))
    initialized = mcp_server.dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert initialized["result"]["serverInfo"]["name"] == "radmeasure"
    listed = mcp_server.dispatch({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert [tool["name"] for tool in listed["result"]["tools"]] == [
        "list_radmeasure_capabilities", "list_available_cases", "analyze_radiograph"
    ]
    called = mcp_server.dispatch({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "analyze_radiograph", "arguments": {"image_id": "demo-foot-001"}}})
    payload = json.loads(called["result"]["content"][0]["text"])
    assert called["result"]["isError"] is False
    assert payload["provenance"]["mode"] == "deterministic_synthetic_demo"
