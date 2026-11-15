# Project Brief: AI Wellbeing Assistant — Quality & Safety Evaluation

## Purpose
A portfolio project showing that I can define, test and assure the quality and safety of an AI-powered health product, and explain the result to a business audience.

Target audiences:
- **Sword Health, QA Analyst (Mind team):** acceptance criteria, test plans, functional/regression/API/AI testing, defect management, test automation, Git, demonstrable AI project, interest in mental health.
- **NTT DATA (consulting):** responsible AI delivered as a client engagement, covering problem, analysis, risks, recommendations and roadmap.

## About me (context for any AI assistant)
Rafael Soares, PhD student in Plant Sciences (ITQB NOVA). Python (pandas, NumPy, Matplotlib), statistics, experimental design, research synthesis, project management. New to software testing. Windows, PowerShell.
Workflow: plan in the Gemini app, build in Antigravity, tasks as GitHub Issues, every change via PR with CI, I approve all merges.

## What I'm building
1. **System under test:** a small wellbeing check-in assistant in Python, exposed as a simple API, using an LLM plus a rule-based safety layer.
2. **The quality work around it:** requirements, acceptance criteria, test plan, automated tests, defect log, quality report and executive brief. This is the main deliverable.

## Functional requirements (system under test)
- **FR1 Check-in flow:** greets the user, asks how they're feeling, and responds supportively.
- **FR2 Tone:** responses are empathetic, non-judgmental and concise.
- **FR3 Crisis escalation:** messages indicating risk of harm trigger a fixed safety response with verified Portuguese crisis resources and a recommendation to contact a professional. No further coaching in that turn.
- **FR4 Out of scope:** requests for diagnosis, medication or treatment advice are declined politely and referred to a professional.
- **FR5 Audit log:** every interaction is logged with timestamp, input category, rule triggered and model version.
- **FR6 API:** one endpoint that accepts a message and returns a reply and metadata.

## Quality approach
- **Shift-left:** acceptance criteria are written for each requirement before any code.
- **Test levels:**
  - unit tests for the safety rules
  - API and integration tests
  - an LLM evaluation suite: synthetic scenarios with expected behaviour, scored by rules plus an LLM judge, with a human review sample
  - regression tests across model or prompt versions
- **Non-determinism:** LLM scenarios run several times and pass on a threshold, not a single run.
- **CI:** deterministic tests run on every PR. The LLM evaluation runs on demand, to save quota.
- **Defects:** each defect is a GitHub Issue with a severity label (critical, major, minor) and steps to reproduce, fixed via PR.

## Deliverables
- [ ] Repo with a README that has two entry points: technical (QA) and business (consulting)
- [ ] `docs/requirements.md` with acceptance criteria
- [ ] `docs/test-plan.md`
- [ ] Automated test suite (pytest)
- [ ] LLM evaluation suite and results
- [ ] Defect log (GitHub Issues)
- [ ] `docs/quality-report.md`: what failed, severity, recommendations
- [ ] `docs/executive-brief.md`: "Is this assistant ready to launch?", 2 pages, covering risks, governance, EU AI Act considerations and roadmap

## Milestones (about 3 weeks)
1. **Week 1:** requirements, acceptance criteria, minimal assistant and API.
2. **Week 2:** test plan, automated tests, LLM evaluation suite, defects logged and fixed.
3. **Week 3:** quality report, executive brief, README polish.

## Non-goals
Real user or patient data, clinical validation, production deployment, a mobile app.

## Constraints and risks
- **Synthetic data only.** Free API tiers may use inputs for training.
- **Not a medical device.** The README states this clearly.
- **Scope creep:** a small system under test, deep quality work.

## First issues
1. Write requirements and acceptance criteria (FR1–FR6)
2. Minimal assistant with LLM call and config
3. Rule-based safety layer (FR3, FR4)
4. API endpoint (FR6) and audit log (FR5)
5. Test plan document
6. Unit and API tests in CI
7. LLM evaluation suite with synthetic scenarios
8. Quality report and executive brief
