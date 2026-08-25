from geomed_copilot.tools import GeoMedTools


class _Predictor:
    identifiers = {"case-1", "case-2"}


class _Service:
    predictor = _Predictor()

    def analyze(self, image_id, question, top_k):
        if image_id not in self.predictor.identifiers:
            raise KeyError(image_id)
        return {
            "image_id": image_id,
            "question": question,
            "top_k": top_k,
            "provenance": {
                "mode": "locked_prediction_artifact_replay",
                "live_encoder_inference": False,
            },
        }


def test_capabilities_are_honest_about_replay_backend():
    tools = GeoMedTools(_Service())
    capabilities = tools.capabilities()
    assert capabilities["available_cases"] == 2
    assert capabilities["accepted_input"] == "image_id"
    assert capabilities["live_encoder_inference"] is False


def test_analyze_validates_arguments_and_preserves_provenance():
    tools = GeoMedTools(_Service())
    result = tools.analyze_radiograph(" case-1 ", " Measure HVA ", 2)
    assert result["image_id"] == "case-1"
    assert result["provenance"]["live_encoder_inference"] is False


def test_analyze_rejects_invalid_arguments():
    tools = GeoMedTools(_Service())
    for image_id, question, top_k in [("", "x", 1), ("x", "", 1), ("x", "y", 0)]:
        try:
            tools.analyze_radiograph(image_id, question, top_k)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError("expected ValueError")
