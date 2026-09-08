from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MeasurementProtocol:
    name: str
    anatomy: str
    required_view: str
    required_entities: tuple[str, ...]
    tools: tuple[str, ...]
    executor: str
    repair_actions: tuple[str, ...]
    maximum_repair_degrees: float

    def to_dict(self) -> dict:
        return asdict(self)


class ProtocolRegistry:
    """Allow-list for every plan and recovery action the agent may execute."""

    def __init__(self, protocols: tuple[MeasurementProtocol, ...] | None = None) -> None:
        configured = protocols or DEFAULT_PROTOCOLS
        self._protocols = {protocol.name: protocol for protocol in configured}

    def get(self, name: str) -> MeasurementProtocol:
        try:
            return self._protocols[name.upper()]
        except KeyError as exc:
            raise ValueError(f"unsupported protocol: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(self._protocols)

    def describe(self) -> list[dict]:
        return [protocol.to_dict() for protocol in self._protocols.values()]

    def validate_tools(self, protocol: str, tools: tuple[str, ...]) -> None:
        allowed = set(self.get(protocol).tools)
        invalid = set(tools) - allowed
        if invalid:
            raise ValueError(f"tools not allowed for {protocol}: {sorted(invalid)}")


FOOT_TOOLS = (
    "landmark_detector",
    "geometry_executor",
    "measurement_validator",
    "retrieve_similar_cases",
    "retrieve_evidence",
)

DEFAULT_PROTOCOLS = (
    MeasurementProtocol(
        name="HVA",
        anatomy="foot",
        required_view="weight_bearing_ap_foot",
        required_entities=("great_toe_axis", "first_metatarsal_axis"),
        tools=FOOT_TOOLS,
        executor="acute_angle",
        repair_actions=("reexecute_from_verified_geometry",),
        maximum_repair_degrees=5.0,
    ),
    MeasurementProtocol(
        name="IMA",
        anatomy="foot",
        required_view="weight_bearing_ap_foot",
        required_entities=("first_metatarsal_axis", "second_metatarsal_axis"),
        tools=FOOT_TOOLS,
        executor="acute_angle",
        repair_actions=("reexecute_from_verified_geometry",),
        maximum_repair_degrees=5.0,
    ),
)
