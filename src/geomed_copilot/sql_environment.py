from __future__ import annotations

import re
import sqlite3
from typing import Any

from .bounded_runtime import ActionProposal


MUTATING = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace)\b", re.I)


class SQLiteRepairEnvironment:
    """Disposable SQL environment with a read-only policy and deterministic verifier."""

    def __init__(self, connection: sqlite3.Connection, expected_columns: tuple[str, ...] = ()) -> None:
        self.connection = connection
        self.expected_columns = expected_columns

    def authorize(self, proposal: ActionProposal) -> tuple[bool, str]:
        if proposal.action.upper() not in {"KEEP", "REPAIR"}:
            return False, "unregistered_action"
        if proposal.tool != "sql_query":
            return False, "unregistered_tool"
        sql = str((proposal.arguments or {}).get("sql", "")).strip()
        if not sql or MUTATING.search(sql) or not sql.lower().startswith("select"):
            return False, "read_only_policy_violation"
        if ";" in sql.rstrip(";"):
            return False, "multiple_statements_forbidden"
        return True, "authorized_read_only_query"

    def execute(self, proposal: ActionProposal) -> dict[str, Any]:
        sql = str((proposal.arguments or {})["sql"])
        cursor = self.connection.execute(sql)
        return {"columns": tuple(item[0] for item in cursor.description), "rows": tuple(cursor.fetchall())}

    def verify(self, proposal: ActionProposal, output: Any) -> tuple[bool, str]:
        if self.expected_columns and tuple(output["columns"]) != self.expected_columns:
            return False, "output_contract_mismatch"
        return True, "execution_and_contract_verified"


def demo_database() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.executescript("""
        CREATE TABLE employees(id INTEGER PRIMARY KEY, name TEXT, department TEXT, salary INTEGER);
        INSERT INTO employees(name, department, salary) VALUES
          ('Ada', 'AI', 150000), ('Grace', 'Systems', 140000), ('Linus', 'Systems', 130000);
    """)
    return connection
