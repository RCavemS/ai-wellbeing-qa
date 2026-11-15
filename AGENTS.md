# AGENTS.md

Instructions for any AI agent working in this repo.

## Project
- What it is: Quality and safety evaluation portfolio project for an AI-powered wellbeing check-in assistant with a rule-based safety layer and simple API.
- Stack: Python, FastAPI, pytest
- Run tests: `pytest`
- Run locally: `python -m src.assistant`

## Environment
- OS: Windows. Shell: PowerShell.
- Use PowerShell syntax. Don't chain commands with `&&`; run them one at a time
  or separate them with `;`.

## Workflow rules
1. Every change starts from a GitHub Issue. Work on a branch named `issue-<number>-<short-name>`.
2. Never push to `main`. Open a pull request that references the issue ("Closes #<number>").
3. Run the tests before opening a PR. Add tests for new behavior.
4. Keep PRs small: one issue per PR.
5. Never read, print or commit `.env` files, keys or other secrets.
6. If something is unclear, ask instead of guessing.

## Skills
Reusable skills live in `.skills/`. Before starting a task, check whether a skill
there matches it (each skill's SKILL.md starts with a description of when to use it).
If one matches, read that SKILL.md and follow it.

## Conventions
- <add rules here whenever an agent makes the same mistake twice>
