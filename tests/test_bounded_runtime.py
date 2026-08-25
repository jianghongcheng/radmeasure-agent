from geomed_copilot.bounded_runtime import ActionProposal, BoundedAgentRuntime
from geomed_copilot.sql_environment import SQLiteRepairEnvironment, demo_database


def test_sql_environment_keeps_verified_read_only_query():
    env = SQLiteRepairEnvironment(demo_database(), ("name",))
    proposal = ActionProposal("REPAIR", "sql_query", {
        "sql": "SELECT name FROM employees WHERE department = 'Systems' ORDER BY name"
    })
    outcome = BoundedAgentRuntime().run(proposal, env)
    assert outcome.decision == "KEEP"
    assert outcome.output["rows"] == (("Grace",), ("Linus",))


def test_sql_environment_stops_mutating_or_unregistered_actions():
    env = SQLiteRepairEnvironment(demo_database())
    for proposal in (
        ActionProposal("REPAIR", "sql_query", {"sql": "DROP TABLE employees"}),
        ActionProposal("REPAIR", "shell", {"sql": "SELECT * FROM employees"}),
    ):
        outcome = BoundedAgentRuntime().run(proposal, env)
        assert outcome.decision == "STOP"
        assert not any(step["step"] == "execute" for step in outcome.trajectory)


def test_sql_environment_stops_tool_error_and_contract_mismatch():
    env = SQLiteRepairEnvironment(demo_database(), ("name",))
    invalid = ActionProposal("REPAIR", "sql_query", {"sql": "SELECT missing FROM employees"})
    assert BoundedAgentRuntime().run(invalid, env).decision == "STOP"
    mismatch = ActionProposal("REPAIR", "sql_query", {"sql": "SELECT salary FROM employees"})
    assert BoundedAgentRuntime().run(mismatch, env).reason == "output_contract_mismatch"
