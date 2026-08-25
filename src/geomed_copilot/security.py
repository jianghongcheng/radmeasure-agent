from __future__ import annotations

import hmac
import json
import os
from dataclasses import dataclass


ROLE_LEVEL = {"viewer": 1, "operator": 2, "admin": 3}


@dataclass(frozen=True)
class Principal:
    name: str
    role: str


class ApiKeyAuthorizer:
    """Small API-key RBAC boundary; raw keys are never logged or persisted."""

    def __init__(self, principals: dict[str, Principal]) -> None:
        if not principals:
            raise ValueError("at least one API key is required")
        self._principals = principals

    @classmethod
    def from_env(cls) -> "ApiKeyAuthorizer":
        raw = os.environ.get("GEOMED_API_KEYS")
        if not raw:
            raise RuntimeError(
                "GEOMED_API_KEYS must be a JSON object mapping API keys to {name, role}"
            )
        rows = json.loads(raw)
        principals = {}
        for key, value in rows.items():
            role = value["role"]
            if role not in ROLE_LEVEL:
                raise ValueError(f"invalid role: {role}")
            principals[key] = Principal(value["name"], role)
        return cls(principals)

    def authenticate(self, api_key: str | None, minimum_role: str) -> Principal:
        if minimum_role not in ROLE_LEVEL:
            raise ValueError(f"invalid minimum role: {minimum_role}")
        matched = None
        if api_key:
            for candidate, principal in self._principals.items():
                if hmac.compare_digest(api_key, candidate):
                    matched = principal
                    break
        if matched is None:
            raise PermissionError("invalid or missing API key")
        if ROLE_LEVEL[matched.role] < ROLE_LEVEL[minimum_role]:
            raise PermissionError(f"{minimum_role} role required")
        return matched
