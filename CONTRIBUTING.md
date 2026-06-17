# Contributing

## Workflow

1. Branch off `main`: `git checkout -b lane/<name>/<topic>`.
2. Put lane code under `lanes/<your_folder>/`. Import shared types from
   `microverse_core`. Do not copy contract definitions into your lane.
3. Before you open a PR: `python scripts/smoke_test.py` prints OK, and `pytest`
   passes.
4. Open a PR using the template. Lance reviews and merges. Reviews are quick;
   keep PRs small and single-topic so they stay quick.

## Changing the contracts

`microverse_core/contracts.py` is a shared interface. Renaming a field or
changing a type there can break four other lanes. If you need a change:

1. Raise it at the all-hands or in the coordination session first.
2. Open the PR with the change plus updated `tests/test_contracts.py`.
3. Tag the lanes that consume the record you are touching.

## Module README rule

Every module gets a `README.md` with three sections, one short paragraph each:
what it does, how to run it, who to ask.

## Commit messages

Specific, present tense, says what changed and why it mattered. The reader is
future-you at 11pm in week 6.
