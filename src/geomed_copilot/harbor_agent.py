from __future__ import annotations

import base64
import json
from pathlib import Path
import shlex

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from .harbor_export import frozen_sql_submission


class RadMeasureFrozenPlannerAgent(BaseAgent):
    """Replay the frozen Qwen3-8B proposals through a Harbor task.

    This agent deliberately performs no new model inference. It allows the exact
    proposals used by the historical 36-case v1 ablation to be evaluated inside an
    isolated Harbor environment without generation randomness.
    """

    @staticmethod
    def name() -> str:
        return "radmeasure-frozen-planner"

    def version(self) -> str:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        return None

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        root = Path(__file__).resolve().parents[2]
        rows = frozen_sql_submission(
            root / "data/benchmarks/sql_repair_v1.json",
            root / "outputs/portfolio/sql_harness_ablation_qwen3_8b.json",
        )
        encoded = base64.b64encode((json.dumps(rows, indent=2) + "\n").encode()).decode()
        command = (
            "python -c "
            + shlex.quote(
                "import base64,pathlib;"
                f"pathlib.Path('/app/submission.json').write_bytes(base64.b64decode('{encoded}'))"
            )
        )
        result = await environment.exec(command=command, cwd="/app", timeout_sec=30)
        if result.return_code != 0:
            raise RuntimeError(f"failed to materialize frozen submission: {result.stderr}")
        context.metadata = {
            "baseline": "frozen_qwen3_8b_proposals",
            "proposal_count": len(rows),
            "generation_replayed": True,
        }


class RadMeasureV3FrozenPlannerAgent(BaseAgent):
    """Replay the preregistered v3 Qwen3-8B proposals in Harbor."""

    @staticmethod
    def name() -> str:
        return "radmeasure-v3-frozen-planner"

    def version(self) -> str:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        return None

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        root = Path(__file__).resolve().parents[2]
        rows = frozen_sql_submission(
            root / "data/benchmarks/sql_repair_v3_confirmatory.json",
            root / "outputs/portfolio/sql_harness_v3_qwen3_8b_confirmatory.json",
        )
        encoded = base64.b64encode((json.dumps(rows, indent=2) + "\n").encode()).decode()
        command = (
            "python -c "
            + shlex.quote(
                "import base64,pathlib;"
                f"pathlib.Path('/app/submission.json').write_bytes(base64.b64decode('{encoded}'))"
            )
        )
        result = await environment.exec(command=command, cwd="/app", timeout_sec=30)
        if result.return_code != 0:
            raise RuntimeError(f"failed to materialize v3 frozen submission: {result.stderr}")
        context.metadata = {
            "baseline": "frozen_qwen3_8b_v3_confirmatory",
            "proposal_count": len(rows),
            "generation_replayed": True,
        }
