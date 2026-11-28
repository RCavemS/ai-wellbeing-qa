# ai-wellbeing-qa

Quality and safety evaluation portfolio project for an AI-powered wellbeing check-in assistant with a hybrid safety layer (deterministic rules + constrained LLM risk classifier), structured JSONL audit logging, and a FastAPI REST endpoint.

## Quickstart

- **Run tests (offline with mocks):**
  ```powershell
  pytest
  ```
- **Start the FastAPI server locally:**
  ```powershell
  uvicorn src.api:app --host 127.0.0.1 --port 8000
  ```
- **Run the interactive CLI locally:**
  ```powershell
  python -m src.assistant
  ```

Built with the AI workflow: see `AGENTS.md`.
