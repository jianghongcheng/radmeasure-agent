#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from geomed_copilot.production import LockedArtifactService


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--evidence-catalog", type=Path, required=True)
    parser.add_argument("--image-id", default="IMG000005.jpg")
    parser.add_argument("--question", default="How should HVA and IMA be measured and audited?")
    args = parser.parse_args()
    service = LockedArtifactService(
        args.predictions, args.annotations, args.split_manifest, args.evidence_catalog)
    print(json.dumps(service.analyze(args.image_id, args.question), indent=2))


if __name__ == "__main__":
    main()
