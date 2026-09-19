# Learning Checkpoint — LangGraph Studio + LangSmith

Status: **designed, not implemented** — blocked on a free LangSmith account/API key that only Abhi can create (see `docs/superpowers/specs/2026-09-19-observability-design.md`). This note captures what was learned from the design/verification pass itself, since that already surfaced real findings before a single line of tracing code exists.

## 1. What problem does this solve that a plain LLM API call / plain cron job doesn't?

`print()`/log statements tell you a node ran; they don't give you a replayable, structured record of *what state it saw and produced*, across a run with four parallel branches and a pause-for-days human-approval gate. The gap this project actually has right now is real: after tonight's build, there is no way to answer "why did run X reject candidate Y" except by reading `agent_memory.db` rows directly — exactly the problem LangSmith tracing exists to solve.

## 2. What did it make easy that was hard/verbose without it?

Nothing yet — not implemented. But the *design* pass made one thing easy: confirming, before writing any code, that tracing itself needs zero code changes (`LANGSMITH_TRACING=true` activates `LangChainTracer` through routing already present in LangGraph's Pregel loop, verified live against the installed `langsmith 0.12.2`). That's a real "this will be cheap once unblocked" finding, not an assumption.

## 3. What did it make hard or awkward?

- **The framing "Studio is free/local, LangSmith needs a key" turned out to be wrong**, and only verification caught it. `langgraph dev`'s Studio link is built against `debugger_origin` defaulting to `https://smith.langchain.com` — Studio's UI is a hosted web app requiring the same LangSmith login LangSmith tracing needs. Both halves of this stack piece are blocked on one account, not two independent capabilities with different costs.
- **`langgraph dev` needs a dependency this project doesn't have** (`langgraph-api` + `langgraph-runtime-inmem`, pulled in only via the `langgraph-cli[inmem]` extra) — real scope beyond a config file, caught by reading the CLI's own source rather than assuming Studio was purely local config.
- **A real, open, unresolved upstream bug** (`langchain-ai/langgraph#6494`) reports Studio hanging on local Agent-server runs on this project's exact Windows build. This means "the server starts" cannot be the completion gate for this phase — "a real run executes end-to-end" has to be, and there's a real chance Studio simply doesn't work reliably on this machine regardless of the account being sorted out. (Notably, the *separate* Temporal phase hit its own real hang on its local test server the same night — a pattern worth taking seriously: local dev-server-style tooling on this Windows machine has now shown two independent hangs.)
- **`pydantic-ai` has no LangSmith integration at all.** Env-var tracing would give node-level spans and timing, but not the per-LLM-call token/cost accounting this project's own cost-control rules actually want — a real gap between "observability is wired" and "observability answers the questions this project cares about."

## 4. When would you reach for this in a real job?

Any agent system complex enough that "which node did what, with what token cost" stops being answerable by reading logs — multi-step/multi-branch graphs, anything with a human-approval or long-pause step where you need to explain a decision after the fact. Also a genuine reminder: verify a tool's actual account/infrastructure requirements from its source/CLI before scoping a phase around an assumption ("local and free") that turned out to be only half true.
