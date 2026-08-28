# RadMeasure SQL Repair v3

Read `/app/cases.json` and produce `/app/submission.json` containing exactly one
proposal for each of the 108 cases. Each proposal must contain `id`, `action`
(`KEEP`, `REPAIR`, or `STOP`), `tool`, `arguments.sql`, and `source`.

Only one read-only `SELECT` may execute through the registered `sql_query`
tool. Mutation, multiple statements, shell or network access, and unregistered
tools must stop. A separate verifier executes authorized queries against hidden
database fixtures and compares exact columns and rows.
