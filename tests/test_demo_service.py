from radmeasure.production import DemoService
from radmeasure.tools import RadMeasureTools


def test_demo_service_is_runnable_and_explicitly_synthetic():
    tools = RadMeasureTools(DemoService())
    capabilities = tools.capabilities()
    assert capabilities["mode"] == "deterministic_synthetic_demo"
    assert capabilities["available_cases"] == 1

    result = tools.analyze_radiograph("demo-foot-001", top_k=2)
    assert result["status"] == "complete"
    assert result["provenance"]["mode"] == "deterministic_synthetic_demo"
    assert result["provenance"]["live_encoder_inference"] is False
    assert len(result["traces"]) == 4
