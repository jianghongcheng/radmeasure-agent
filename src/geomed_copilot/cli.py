from __future__ import annotations

import argparse
import json

from .models import CopilotRequest
from .orchestrator import GeoMedCopilot
from .retrieval import CaseRetriever, HybridRetriever
from .sample_data import CASES, DEMO_LANDMARKS, EVIDENCE


def build_demo() -> GeoMedCopilot:
    return GeoMedCopilot(HybridRetriever(EVIDENCE), CaseRetriever(CASES))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline GeoMed Copilot demo")
    parser.add_argument("--question", default="How should HVA be measured and verified?")
    args = parser.parse_args()
    request = CopilotRequest(
        question=args.question,
        image_id="demo-hva-001",
        landmarks=DEMO_LANDMARKS,
        predicted_angles={"HVA": 15.2, "IMA": 8.1},
    )
    print(json.dumps(build_demo().run(request).to_dict(), indent=2))


if __name__ == "__main__":
    main()

