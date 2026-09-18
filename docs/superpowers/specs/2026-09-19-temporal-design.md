# Temporal Durable Daily Monitor — Design Spec

**Status:** Draft for review (designed 2026-09-19; **design only — nothing implemented, nothing installed**). Needs Abhi's sign-off on the Open Decisions — especially the new dependency, the notification channel, and the run cadence — before planning.
**Phase:** 10 of the Portfolio Advisor Fleet build (see `plans/plan.md`) — the last major piece of the Plan A stack.
**Depends on:** Phase 3 (`checkpointer.py`), Phase 7 (`store.py`, the fail-loud `store is None` guard in all three new-buy specialists), and **Phase 8** (`build_compiled_graph(*, checkpointer, store)`, the `human_approval_gate_node` `interrupt()`, and the "no `Command(resume` anywhere in `src/`" prohibition). Phase 8 is currently a design spec only — this phase cannot be implemented before it lands.
**Resolves:** the Phase 7 spec's open risk — *"SQLite concurrency is verified single-process only… the Temporal phase could introduce two overlapping processes writing the same file — flagged as something that phase must serialise or re-verify."* See **Architecture 3**.

## Goal

`plan.md`'s fifth stack piece: *"A Temporal workflow that runs daily … invokes the LangGraph advisor … and notifies you — built so a crash or restart mid-run resumes cleanly rather than duplicating work or losing state."*

Three things this phase must deliver, and one it must **not**:

1. A durable, scheduled trigger that survives worker crashes, machine restarts, and mid-run failures.
2. A single, serialised, lock-protected path into the graph, so a scheduled run and a manual run can never write `agent_memory.db` at the same time.
3. A notification when a run finishes — in particular when it is **waiting for Abhi's approval**.
4. **It must never approve anything.** A scheduled run that reaches the human-approval gate leaves it pending, forever, and tells a human. This is the load-bearing guardrail of the phase.

## Scope

**In scope:**

- `src/agentic_frameworks/runner.py` (new) — `run_advisor_once(thread_id)`, the **single** function in the codebase that compiles and invokes the advisor graph, plus its `python -m` CLI entry point. Both the Temporal activity and any manual run go through it.
- `src/agentic_frameworks/db_lock.py` (new) — the process-level advisory lock (stdlib only) that makes concurrent SQLite writers impossible.
- `src/agentic_frameworks/graph/checkpointer.py`, `store.py` (modified, small) — each acquires the lock before opening `agent_memory.db`. This is where the guarantee becomes structural rather than conventional (Architecture 3).
- `src/agentic_frameworks/temporal/shared.py` (new) — stdlib dataclasses crossing the workflow/activity boundary.
- `src/agentic_frameworks/temporal/activities.py` (new) — three activities: check-pending, run-graph, notify.
- `src/agentic_frameworks/temporal/workflows.py` (new) — one workflow, `PortfolioAdvisorDailyWorkflow`.
- `src/agentic_frameworks/temporal/worker.py` (new) — the worker process entry point.
- `src/agentic_frameworks/temporal/schedule.py` (new) — a one-shot script that creates/updates the daily Schedule.
- `pyproject.toml` (modified) — **one new dependency, `temporalio`.** Real scope; see Background.
- Tests per Testing below.

**Out of scope (explicitly deferred, with reasons):**

- **Approving a pending run.** The approval *interface* (a CLI that reads the pending interrupt and resumes it with a real human decision) belongs with Phase 8's contract, not here. This phase only *detects and reports* pendency. Nothing here can resume anything.
- **Any real delivery channel** (email, Telegram, Slack, push). Open Decision 2 — no credentials are configured, no message is sent anywhere, and nothing will be until Abhi picks a channel. Local-only reporting is what ships by default.
- **`temporalio.contrib.langgraph` (`LangGraphPlugin`).** Rejected on verified technical grounds, not on effort — see Architecture 1. Revisit only if the Store limitation is lifted upstream.
- **Deployment / running the worker as a Windows service.** `plan.md`'s Deployment section keeps P1 local-only; a hand-started worker plus a hand-started dev server is the P1 shape. Auto-start is a P2 question.
- **Temporal Cloud.** Not needed and not used — see Local Setup.
- **Alert-rule interpretation.** `plan.md`'s Temporal sketch also mentions "checks alert rules / daily performance data … to interpret anomalies." The `portfolio_data_mcp` server does expose `fired_alerts`/`daily_performance`, but no node in this graph reads them, and adding an anomaly-interpretation node is a *graph* phase, not a scheduling phase. Named here so it isn't quietly dropped; see Open Decision 5.
- **Studio/LangSmith** — Phase 9, currently blocked on a LangSmith account (`phase-9-observability-blocked`). Temporal's own Web UI gives a *different* kind of observability (workflow history, retries, timers) and is not a substitute.

## Background / Constraints (verified 2026-09-19 — what was actually checked, and how)

### `temporalio` is **not installed**. This phase is a new dependency.

Verified three ways: `temporalio` is absent from `.venv/Lib/site-packages`; `grep -i temporal pyproject.toml uv.lock` returns nothing; and `.venv\Scripts\python.exe -c "import temporalio"` raises `ModuleNotFoundError`. There is also no `temporal` CLI on `PATH`.

So: **nothing about Temporal is available today.** Adding it is real scope — `uv add temporalio`, a lockfile change, and a ~15 MB wheel. Per the "no silent scope growth" rule this is flagged as a decision, not folded in (Open Decision 1). **No package was installed while writing this spec.**

Verified against the live PyPI metadata (not from memory):

- Latest `temporalio` is **1.33.0**, `requires_python >=3.10`, Apache-2.0, published by the Temporal team.
- Its only hard runtime deps are `nexus-rpc`, `protobuf`, `types-protobuf`, `typing-extensions` — small, no LLM/DB deps, no conflict with the installed `langgraph 1.2.11` / `pydantic-ai 2.42.0` stack.
- **Wheels are `cp310-abi3`, including `temporalio-1.33.0-cp310-abi3-win_amd64.whl`.** This project's venv is CPython **3.14.0 on win_amd64**. `abi3` means the prebuilt binary (Temporal's Rust core) works on 3.14 with **no compiler and no source build** — the one thing that could have made this dependency genuinely expensive on Windows, and it isn't.
- `uv add temporalio` (no extras) is sufficient. The `pydantic` extra is **not** needed under the dataclass-only boundary this design uses (Components).

### Free local setup — verified, no account, no Cloud

**Temporal Cloud is not involved at any point.** Two free local options exist; this design uses the second for development and offers the first for day-to-day running:

1. **The `temporal` CLI dev server** — `temporal server start-dev`, from https://github.com/temporalio/cli (MIT, no login). A single self-contained binary with SQLite persistence and a bundled Web UI (default gRPC `127.0.0.1:7233`, UI `127.0.0.1:8233`). Not currently installed; would be a manual download/`winget`/`scoop` step.
2. **A Python-only path with no manual CLI install at all** — `temporalio.testing.WorkflowEnvironment.start_local()`. Read directly from the wheel's source: *"Internally, this uses the Temporal CLI dev server from https://github.com/temporalio/cli. This is a self-contained binary for Temporal using Sqlite persistence. This call will download the CLI to a temporary directory by default if it has not already been downloaded."* Its verified keyword arguments include `ui: bool = False`, `ui_port`, `port`, `dev_server_database_filename` (*"Unset default means only in-memory Sqlite"*), `dev_server_existing_path` (skip the download if the binary is already present), and `dev_server_download_ttl`.

**What this actually costs, stated plainly:** one **internet download** of the CLI binary (~tens of MB) on first use, cached thereafter. It is free and login-free, but it is *not* offline-capable on first run. An offline machine needs the binary fetched once and then `dev_server_existing_path=` pointed at it. There is **no** pure-Python, dependency-free embedded Temporal server — the server is always that binary.

**Schedules** (`temporalio.client.Client.create_schedule`, `ScheduleSpec.cron_expressions`, `ScheduleOverlapPolicy`) are all present in the SDK and are a server feature the dev server carries. *Not yet executed live* — confirming `temporal schedule list` works against the dev server is the first task of implementation, not an assumption of this design.

### `temporalio.contrib.langgraph` exists — and is verifiably wrong for this project

Version 1.33.0 ships an official, **experimental** LangGraph plugin (`temporalio/contrib/langgraph/`, installed via the `langgraph` extra) that runs each LangGraph node as its own Temporal Activity. It looks like the obvious answer. It is not, and the reasons are verified in its own shipped source and README, not guessed:

- **"Stores are not supported."** Verbatim from `temporalio/contrib/langgraph/README.md`: *"LangGraph's `Store` … isn't accessible inside Activity-wrapped nodes: the Store holds live state that can't cross the Activity boundary … If you pass a store, the plugin logs a warning on first use and `runtime.store` is `None` inside nodes."* Confirmed in `_activity.py`, which strips the runtime's non-serializable fields (*"it holds non-serializable stream_writer, store"*) and emits a once-per-run warning. **Consequence for this codebase:** Phase 7 put an explicit `if store is None: raise ValueError(...)` at the top of all three new-buy specialists precisely so a missing store can never silently disable the cost control. Under this plugin, that guard fires — the graph crashes in three nodes. The plugin's suggested remedy ("use a backend-backed store — Postgres/Redis") is ruled out by the `agent_memory.db`-only guardrail.
- **It tells you to use `InMemorySaver`.** Verbatim: *"If your LangGraph code requires a checkpointer (for example, if you're using interrupts), use `InMemorySaver`. Temporal handles durability."* That throws away Phase 3's `AsyncSqliteSaver` and, worse, makes the Phase 8 approval interrupt live inside Temporal workflow state — where the *only* way to answer it is a Temporal signal, i.e. exactly the machine-reachable resume path this phase must not create.
- **It is marked experimental** in its own README's first line.
- **It duplicates work Phase 3 already did.** Per-node durability is what the checkpointer provides. Buying it again at the cost of the store and the checkpointer is a bad trade.

**Decision: reject the plugin.** The graph is invoked as a whole, inside one coarse Activity. Documented here rather than silently avoided, because "there is an official integration and we didn't use it" is exactly the kind of choice that must not be invisible.

### Determinism and the workflow sandbox

Temporal workflow code is replayed from history and must be deterministic; **all I/O lives in Activities**. The SDK additionally runs workflow code in an import sandbox (`temporalio/worker/workflow_sandbox/`). Practical consequence for the module layout below: `workflows.py` must import nothing heavy — no `langgraph`, no `pydantic_ai`, no MCP client, no `sqlite3`. It imports `temporalio`, `datetime`, and the stdlib dataclasses in `shared.py` (the latter via `workflow.unsafe.imports_passed_through()`, the standard pattern). Everything expensive is imported inside `activities.py`, which is ordinary Python with no restrictions.

### What the existing code requires to run the graph at all

From `phase-7-langgraph-store`: `graph.py` returns an **uncompiled** `StateGraph`; the caller must open `get_checkpointer()` (an `AsyncSqliteSaver` on `sandbox-data/agent_memory.db`) **and** `get_memory_store()` (an `AsyncSqliteStore` on the same file), compile with both, and `ainvoke` with `config={"configurable": {"thread_id": ...}}`. Phase 8 replaces the ad-hoc version of this with `build_compiled_graph(*, checkpointer, store)` — keyword-only and required — which is what this phase's runner will call. Nodes reach three MCP servers as **stdio subprocesses** (`portfolio_data_mcp` from the sibling project, `uvx yfmcp@latest`, and this project's own `mfapi_mcp`), and LLM nodes need `OPENROUTER_API_KEY` from `.env`. All of that is ordinary async I/O and subprocess work — legal in an Activity, illegal in a Workflow.

## Architecture

Five ideas, in order.

### 1. One coarse Activity runs the whole graph; the Workflow only orchestrates

```
Temporal Schedule (cron, overlap=SKIP)
        │
        ▼
PortfolioAdvisorDailyWorkflow                    ← deterministic, no I/O, no LLM, no DB
        │
        ├─ Activity: check_pending_approvals     ← cheap, read-only on agent_memory.db
        ├─ Activity: run_advisor_graph           ← THE graph run: MCP subprocesses, LLM calls, SQLite
        └─ Activity: notify                      ← writes the pending/completed record
```

The Workflow's entire job is: pick today's `thread_id`, decide whether to run, run, branch on the outcome, notify, finish. That is deterministic branching over Activity results — which is precisely what Temporal replay is good at.

`run_advisor_graph` does **not** re-implement the graph run. It calls `runner.run_advisor_once(...)` — the same function a human calls from the command line. One code path, two triggers. That property is what makes the lock in Architecture 3 airtight and what keeps this phase from quietly forking the system's behaviour into "the scheduled version" and "the manual version."

**What is lost by not decomposing per node:** Temporal's UI won't show one box per graph node, and a failed specialist retries the whole graph rather than just itself. The first is LangSmith's job (Phase 9). The second is mitigated by the checkpointer (Architecture 4) — a retried run resumes after the last completed node, so "retry the activity" is *not* "re-pay for the whole graph."

### 2. The scheduled run cannot approve anything — four structural reasons

Phase 8's spec states the prohibition and pins it with a test: *"no module under `src/agentic_frameworks/` may construct `Command(resume=...)`. Testing item 9 greps the source tree for it. The Temporal phase inherits this: a scheduled run that finds a thread in pending-approval must leave it pending and notify — never resume it."*

This design inherits it by **capability absence**, not by instruction:

1. **The Temporal package lives under `src/agentic_frameworks/temporal/`, so Phase 8's existing grep test covers it the moment the files exist** — no new enforcement mechanism, no chance of the Temporal code sitting outside the check. The test is strengthened here to also assert that no module in that package imports `Command` from `langgraph.types` at all (you cannot call what you never imported), and that `notify_activity`'s signature has no resume-value parameter.
2. **`run_advisor_once()` only ever calls `ainvoke`/`astream` with a plain input dict — never with a `Command`.** Its signature has no parameter that could carry a resume value, so no caller (Temporal or otherwise) can inject one. Resuming a pending thread is a *different* function that this phase does not write.
3. **The Workflow has no signal handler and no query that carries a decision.** A human cannot accidentally approve via the Temporal UI either, because there is nothing there to send a decision *to*. The only way to answer the gate is the (future, Phase-8-contract) human-facing approval path.
4. **The Workflow's reaction to `status == "pending_approval"` is to notify and complete.** It does not wait, does not poll, does not time out into a default. There is no "approve after N hours" branch anywhere, and `HumanApproval` (Phase 8) has no default for `decision` and no default for `approved_by`, so even a malformed injection fails validation rather than defaulting to approved.

Defence in depth, unchanged from Phase 8: approval finalises an advisory *report*, and no order-placement tool exists anywhere in this fleet. Worst case of a hypothetical bug is a marked-up document, never a trade.

### 3. The SQLite-concurrency fix: an OS advisory lock, taken where the database is opened

This is Phase 7's flagged risk, and it is the part of this design that deserves the most scrutiny.

**First, what the risk actually is — stated precisely, not inflated.** SQLite is process-safe; `agent_memory.db` is in WAL mode, which permits many readers and one writer across processes. Two concurrent advisor runs would *not* corrupt the file. The real damage is (a) `database is locked` write-timeout failures under two long-running writers, which is an unverified regime here, and (b) something worse than an error: **two runs interleaving logically** — racing each other's Phase 7 rejection-memory writes, duplicating paid LLM work on the same candidates, and potentially producing two divergent reports for the same day. (b) is the reason to serialise even if (a) never fires.

**Mechanism chosen: a stdlib OS advisory file lock on `sandbox-data/agent_memory.lock`, acquired inside `get_checkpointer()` and `get_memory_store()` themselves.**

- `msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)` on win32, `fcntl.flock(fd, LOCK_EX | LOCK_NB)` on posix. Both stdlib — **no new dependency** beyond `temporalio` itself.
- Process-wide and **re-entrant by refcount**: one process opening both the checkpointer and the store (which every run does) acquires once and releases when the last holder exits. A *second* process fails to acquire.
- On contention: retry with a short bounded backoff (a few seconds — a genuine edge overlap shouldn't hard-fail), then raise `AdvisorMemoryLockedError`. Fail loud, per the standing rule. No silent skip.
- **Scoped to the real `AGENT_MEMORY_DB_PATH` only.** Tests that point at a temp SQLite file are unaffected; the tests that deliberately exercise the real path (`test_store.py`, `test_checkpointer.py`) simply take the lock like anything else — which is correct behaviour, not a workaround.

**Why the lock is taken at the database-open site, not at the entry point.** An entry-point lock (in `runner.py`) would be defeated by any ad-hoc script — and this project *has* one: `HANDOFF.md` currently documents a `python -c "..."` snippet that opens `get_checkpointer()`/`get_memory_store()` directly for Phase 7's manual verification. Putting the lock in `runner.py` would leave exactly that snippet unprotected. Putting it inside `get_checkpointer()`/`get_memory_store()` means **every** path that can possibly open `agent_memory.db` — the Temporal activity, `python -m agentic_frameworks.runner`, the HANDOFF snippet, a future approval CLI, a stray REPL — is serialised, because there is no other way to open those two databases. That is the difference between a convention and a structural guarantee (principle 7).

**Why Temporal's own mutual exclusion is not sufficient, explicitly.** `ScheduleOverlapPolicy.SKIP` (verified present: *"Don't start anything"* when one is already running) and a fixed workflow ID with `WorkflowIDReusePolicy.REJECT_DUPLICATE` both prevent **two Temporal workflow instances**. Neither knows a `python -m agentic_frameworks.runner` process exists — that run never touches the Temporal server at all. Temporal's dedup lives in the server process; the manual run lives in a different process with a different lifecycle. **The only layer both share is the operating system**, which is why the lock has to be an OS lock. `ScheduleOverlapPolicy.SKIP` is still used, as a cheap first layer that avoids even *starting* a doomed second workflow, but it is explicitly not the safety mechanism.

**Why an advisory OS lock and not a sentinel/PID file or a SQLite lock row.** A `O_CREAT|O_EXCL` sentinel or a lock row survives a crashed process — a durable-execution phase that introduces a poison lock requiring manual cleanup after every crash would be self-defeating, and PID-liveness checks on Windows are their own mess. `msvcrt`/`fcntl` locks are **released by the OS when the holding process dies**, crash included, so stale locks cannot exist. A `BEGIN EXCLUSIVE` transaction on `agent_memory.db` was also considered and rejected: held for a whole run it would block the checkpointer's own writes to that same file.

**Honest residual gaps** (no mechanism claimed that isn't real):

- A second process that opens `agent_memory.db` with a **raw `sqlite3`/`aiosqlite` connection**, bypassing `get_checkpointer()`/`get_memory_store()`, is not blocked. Mitigation: a grep test asserting no module under `src/` opens `AGENT_MEMORY_DB_PATH` outside those two functions (the same enforcement style as the `Command(resume` test). Current code already complies.
- The lock protects `agent_memory.db`, **not** `portfolio.db` — which needs no protection, because every connection to it is `mode=ro` and read-only concurrency is exactly what SQLite is good at.
- The lock says nothing about two runs on *different machines* against a shared file. Not applicable at P1 (local-only) and out of scope.
- Verification is owed, not assumed: the Phase 7 spec asked this phase to "serialise **or re-verify**." This design serialises **and** the test plan re-verifies (Testing 5), because a claim like this shouldn't rest on reasoning alone.

### 4. Two durability layers, meeting at one seam: `thread_id`

This is the part most likely to be mistaken for duplicated Phase 3 work, so the boundary is drawn explicitly:

| | **Temporal (this phase)** | **LangGraph checkpointer (Phase 3)** |
|---|---|---|
| Unit of durability | The *orchestration*: did today's run happen, was it retried, was the notification sent | The *graph's progress*: which nodes have completed, with what state |
| Mechanism | Workflow event history + deterministic replay on the server | Checkpoints written per superstep to `agent_memory.db` |
| Survives | Worker crash, machine reboot, activity failure, the whole Python process disappearing mid-run | A crash *within* a run, by resuming a `thread_id` |
| Does **not** cover | Anything *inside* an Activity — a crashed Activity is retried **from its start**, not from mid-execution | Whether the run was ever started, or started twice, or its failure noticed by anyone |
| Analogy | A cron that cannot forget, cannot double-fire, and remembers every past attempt | A savepoint inside the job |

**The seam.** The Workflow derives a deterministic `thread_id` from the scheduled date (`f"advisor-daily-{workflow.now().date().isoformat()}"` — `workflow.now()` is replay-stable, `datetime.now()` is not and would be a determinism bug) and passes it into the Activity. On an Activity retry, Temporal passes the **same** `thread_id`, so LangGraph resumes after the last completed node instead of re-running Research, Risk Analyst, and however many specialists already finished — i.e. **it does not re-pay for completed LLM work.** That single shared identifier is the entire integration between the two layers, and it is what makes `plan.md`'s verification item — *"The Temporal workflow survives a simulated crash/restart mid-run without duplicating work"* — literally true rather than aspirational.

Neither layer is redundant: without Temporal, a crashed run is simply never noticed; without the checkpointer, every retry starts from zero and re-spends money.

**Activity timeouts and retries, concretely:** `start_to_close_timeout` of ~45 minutes (the graph makes many LLM and MCP calls; the real end-to-end duration must be measured before this number is fixed), `heartbeat_timeout` of ~2 minutes with `activity.heartbeat(<node name>)` emitted per chunk from `astream` — which also surfaces live per-node progress in the Temporal UI for free, and lets a dead worker be detected in minutes instead of at the 45-minute timeout. `RetryPolicy(maximum_attempts=3)` with `AdvisorMemoryLockedError` **retryable** (an overlapping manual run will end) and configuration/credential errors **non-retryable** (a missing `OPENROUTER_API_KEY` will not fix itself on attempt 2).

### 5. Notification: local-first, into `agent_memory.db`, nothing sent anywhere

`plan.md` lists the notification channel as an explicit open item (*"Notification channel for the Temporal build (email/Slack/console) — decide when we reach that build"*). This is that moment, and the answer here is deliberately the least presumptuous one.

**What ships by default — three local surfaces, no new service, no credentials, no network:**

1. **The authoritative record goes into `agent_memory.db`**, via the LangGraph Store in a new `("pending_approvals", "advisor")` namespace (key = `thread_id`), holding `status`, `thread_id`, `created_at`, the interrupt's decision-request payload, and a one-line summary. Chosen over a JSON/log file specifically because `CLAUDE.md` says **agent writes go to `agent_memory.db` only** — a new notification *file* would be a new write surface that rule doesn't cover, and it is not this phase's place to quietly widen it (Open Decision 3 asks whether Abhi wants a human-readable file anyway).
2. **The worker's console output** — a clear, human-readable line per run outcome. stdout, not a file.
3. **The Temporal Web UI** (`localhost:8233` when the dev server runs with the UI) — shows the workflow's status, history, retries, and the returned outcome, for free.

**What does not ship, and why:** no email, no Telegram/Slack/Discord bot, no Windows toast, no ntfy.sh. Each needs a real choice from Abhi (an SMTP/app password, a bot token, a hosted topic) and each is a new outbound dependency with its own failure mode and its own secret to leak. **Nothing in this design sends a message anywhere.** Open Decision 2 lists the realistic options with what each actually costs to set up; a channel gets built only after Abhi picks one.

**The honest limitation of local-first:** a pending approval sits in `agent_memory.db` and in the worker's console until Abhi looks. That is strictly better than today (Phase 8's own risk list: *"A pending approval is an open thread with no owner yet… nothing in this phase notifies anyone"*), and strictly worse than a push notification. It is a deliberate floor, not the finished answer.

## Components

### `src/agentic_frameworks/db_lock.py` (new)

`advisor_memory_lock()` — a re-entrant, process-wide context manager over an OS advisory lock on `sandbox-data/agent_memory.lock`; `AdvisorMemoryLockedError`. Stdlib only (`msvcrt` on win32, `fcntl` on posix), selected once at import.

### `src/agentic_frameworks/graph/checkpointer.py` / `store.py` (modified — small)

Each wraps its existing `from_conn_string(...)` in `advisor_memory_lock()`. ~3 lines each. No signature changes, so every existing call site and test keeps working unchanged.

### `src/agentic_frameworks/runner.py` (new)

```
async def run_advisor_once(*, thread_id: str, progress: Callable[[str], None] | None = None) -> AdvisorRunOutcome
```

The one place the advisor graph is compiled and invoked. Opens checkpointer + store, calls Phase 8's `build_compiled_graph(checkpointer=..., store=...)`, drives it with `astream` (so `progress(node_name)` can be forwarded to `activity.heartbeat`), then inspects the final result and `aget_state()` to classify the outcome. **No parameter can carry a resume value; `Command` is never imported.** Also exposes a `python -m agentic_frameworks.runner` CLI for manual runs — which is what `HANDOFF.md`'s ad-hoc `python -c` snippet should be replaced with, so the documented manual path and the scheduled path are the same code.

### `src/agentic_frameworks/temporal/shared.py` (new)

Stdlib `@dataclass`es only — safe to import into the workflow sandbox and serializable by Temporal's **default** JSON converter, so no `pydantic` extra and no custom data converter are needed.

```
AdvisorRunRequest:  thread_id: str
AdvisorRunOutcome:  status: str              # "completed" | "pending_approval" | "failed_lock"
                    thread_id: str
                    summary: str             # one human-readable line
                    pending_payload: dict | None   # the interrupt's decision request, JSON-safe
                    new_buy_count: int
                    action_count: int
Notification:       kind: str, thread_id: str, summary: str, payload: dict | None
DailyRunResult:     status: str, thread_id: str, summary: str
```

### `src/agentic_frameworks/temporal/activities.py` (new)

- `check_pending_approvals(_: None) -> list[str]` — opens the store read-only, returns `thread_id`s in the `("pending_approvals", "advisor")` namespace still marked pending. Cheap; no LLM, no MCP.
- `run_advisor_graph(req: AdvisorRunRequest) -> AdvisorRunOutcome` — calls `run_advisor_once(thread_id=req.thread_id, progress=activity.heartbeat)`.
- `notify(n: Notification) -> None` — writes/updates the store record; prints the human-readable line. Idempotent on `(thread_id, kind)`, because an Activity can legitimately run twice.

### `src/agentic_frameworks/temporal/workflows.py` (new)

`PortfolioAdvisorDailyWorkflow` — one `@workflow.run`, no signals, no queries, no `Command`, no I/O.

### `src/agentic_frameworks/temporal/worker.py` / `schedule.py` (new)

`worker.py`: connect to `localhost:7233`, register the one workflow and three activities on task queue `portfolio-advisor`, run until interrupted. `schedule.py`: create-or-update the `portfolio-advisor-daily` Schedule with `ScheduleSpec(cron_expressions=[...])` and `SchedulePolicy(overlap=ScheduleOverlapPolicy.SKIP)`; idempotent, safe to re-run.

## Data Flow

**`PortfolioAdvisorDailyWorkflow.run() -> DailyRunResult`**

1. `thread_id = f"advisor-daily-{workflow.now().date().isoformat()}"` — deterministic, replay-stable, and the seam with the checkpointer.
2. `pending = await workflow.execute_activity(check_pending_approvals, ...)` (short timeout, 3 attempts).
3. **Reject-early (principle 8):** if `pending` is non-empty → `notify(kind="still_pending")` and return `status="skipped_prior_pending"`. No LLM spend. Rationale and the alternative are Open Decision 4.
4. `outcome = await workflow.execute_activity(run_advisor_graph, AdvisorRunRequest(thread_id), start_to_close_timeout=~45m, heartbeat_timeout=~2m, retry_policy=...)`.
5. Branch on `outcome.status`:
   - `"pending_approval"` → `notify(kind="needs_approval", payload=outcome.pending_payload)`; return. **The workflow ends here. It never resumes the thread.**
   - `"completed"` → `notify(kind="completed")`; return.
   - `"failed_lock"` → surfaces as a retryable Activity failure first; if attempts are exhausted, `notify(kind="skipped_locked")`; return.
6. Return `DailyRunResult` — visible in the Temporal UI and in the Schedule's recent-actions list.

**`run_advisor_once(thread_id)`**

1. `async with get_checkpointer() as cp, get_memory_store() as store:` — **both take the OS lock**; a concurrent manual run raises `AdvisorMemoryLockedError` here, before any MCP subprocess is spawned or any token is spent.
2. `compiled = build_compiled_graph(checkpointer=cp, store=store)` (Phase 8).
3. `async for chunk in compiled.astream({}, config={"configurable": {"thread_id": thread_id}}):` → `progress(node_name)` per chunk.
4. Classify: `"__interrupt__"` present in the final result (and/or `(await compiled.aget_state(config)).next == ("human_approval_gate",)`) → `status="pending_approval"`, with the interrupt's JSON-safe payload attached. Otherwise `status="completed"`, with counts pulled from `recommendation`.
5. Return the dataclass. **No `Command`, no resume, no approval, ever.**

## Error Handling

Fail loud, no `try`/`except` swallowing — the standing rule, and doubly important here because Temporal's retries can make a swallowed error look like success.

- **Lock contention → `AdvisorMemoryLockedError`**, retryable (bounded). Never a silent skip, never "run anyway."
- **Missing `OPENROUTER_API_KEY` / unreachable MCP server / bad config → non-retryable Activity failure.** Retrying a misconfiguration three times just wastes 45-minute timeouts; the workflow fails visibly in the UI.
- **A node raising inside the graph** propagates out of `astream`, fails the Activity, and Temporal retries with the same `thread_id` — so the retry resumes after the last completed node rather than restarting. This is the designed interaction, not an accident.
- **A malformed/absent interrupt payload** raises rather than being coerced into "completed." Misreporting a pending approval as done is the single worst failure mode this phase could have.
- **`notify` is idempotent**, because Temporal guarantees at-least-once Activity execution. A duplicate notification must not produce a duplicate or conflicting record.
- **`portfolio.db` is never opened by any code in this phase** except transitively, read-only, through the graph's existing `mode=ro` MCP path. Nothing here writes to it. Mtime is asserted unchanged by the test suite, as in every prior phase.

## Guardrail check (`CLAUDE.md`, explicitly walked)

- **Advisory only, never execution** — no broker API, no order tool, no new outbound capability at all. The workflow's terminal action is "write a record and stop."
- **Long-term investing only** — a *daily* cadence is monitoring frequency, not trading frequency. Nothing here introduces intraday data, timing signals, or short-horizon logic; the graph it invokes is the same long-horizon advisor. (Open Decision 6 asks whether daily is even the right cadence — weekly may serve the same purpose at a fifth of the cost.)
- **`portfolio.db` read-only, always** — untouched; `portfolio_data_mcp/server.py` not edited.
- **Agent writes go to `agent_memory.db` only** — the notification record goes into the Store in that same file; the one new file is a zero-byte `.lock` sentinel under `sandbox-data/` (no agent data in it), and a human-readable log file is *asked about*, not assumed (Open Decision 3).
- **Human-in-the-loop at consequential decision points** (principle 9) — strengthened, not weakened: this is the first phase where a machine could plausibly have approved something, and it structurally cannot.
- **Guardrails are structural, not instructional** (principle 7) — the no-auto-approve guarantee is capability absence plus a grep test; the concurrency guarantee is an OS lock at the database-open site, not a documented convention.

## Local Setup (what a fresh machine actually needs)

1. `uv add temporalio` — one dependency, prebuilt `abi3` wheel, no compiler (Open Decision 1).
2. A dev server, either:
   - `temporal server start-dev --db-filename sandbox-data/temporal-dev.db --ui-port 8233` after a one-time CLI install, **or**
   - a small `scripts/temporal_dev_server.py` calling `WorkflowEnvironment.start_local(ui=True, port=7233, dev_server_database_filename=...)`, which downloads and caches the same binary automatically — **no manual install, no account, no Cloud**.
   Either way: the first run needs internet to fetch the binary; afterwards it is local and offline. **Use `dev_server_database_filename` — the default is in-memory, which would throw away the Schedule on every restart.**
3. `python -m agentic_frameworks.temporal.worker` in one terminal.
4. `python -m agentic_frameworks.temporal.schedule` once, to create the daily Schedule.
5. Manual run, any time: `python -m agentic_frameworks.runner` (serialised against the scheduled run by the lock).

## Testing (Phase gate — must pass before this phase counts as done)

Project convention: real dependencies wherever free, `TestModel`/`FunctionModel` wherever a real call would cost money, `portfolio.db` never mocked, no fabricated tickers anywhere.

1. **The lock, cross-process, for real.** Hold `advisor_memory_lock()` in the test process, then `subprocess` a second Python process that calls `get_checkpointer()` — assert it raises `AdvisorMemoryLockedError` and does **not** open the database. Then release and assert the subprocess succeeds. This is the test that actually retires Phase 7's flagged risk; nothing else in the suite proves it.
2. **The lock is re-entrant within one process** — `get_checkpointer()` and `get_memory_store()` open together (as every real run does) without deadlocking; and the lock is released after a crash (kill a lock-holding subprocess, assert the next acquire succeeds — the property a PID-file design would fail).
3. **No raw opener bypasses the lock** — grep test: no module under `src/` opens `AGENT_MEMORY_DB_PATH` outside `checkpointer.py` / `store.py`.
4. **No auto-approval, structurally** — extend Phase 8's existing `Command(resume` grep to explicitly cover `src/agentic_frameworks/temporal/` and `runner.py`; assert no module there imports `Command`; assert `run_advisor_once`'s signature has no resume-shaped parameter; assert `PortfolioAdvisorDailyWorkflow` defines no `@workflow.signal`.
5. **The pending-approval path, end to end, without Temporal** — drive the real graph (real `portfolio.db`, real temp-file checkpointer + store, `TestModel` patched into the LLM nodes) down the approval path via `run_advisor_once`; assert `status == "pending_approval"`, a JSON-serializable payload, and — critically — that the thread is **still pending** afterwards (`aget_state().next == ("human_approval_gate",)`). The proof that the runner observed the gate without answering it.
6. **The workflow's branching, deterministically** — `WorkflowEnvironment.start_local()` (free, per Local Setup) with the **real workflow** and **mocked activities**: `run_advisor_graph` returning `pending_approval` → exactly one `notify(kind="needs_approval")` and no second graph run; returning `completed` → the completed path; `check_pending_approvals` returning a non-empty list → `run_advisor_graph` is **never called**. Mocking activities here is not a compromise — it is how Temporal workflow logic is meant to be tested, and the real graph path is covered by test 5.
7. **Crash/restart durability, genuinely simulated** (`plan.md`'s own verification item) — a failing-then-succeeding activity mock proves Temporal retries and completes; and separately, a real-graph run interrupted mid-way then re-invoked with the **same `thread_id`** must resume rather than re-run completed nodes (assert via node call counters that Research/Risk Analyst do not execute twice). That second half is what proves the two durability layers actually compose.
8. **Deterministic `thread_id`** — the same scheduled date always yields the same id; `workflow.now()` is used, never `datetime.now()` (Temporal's replayer will fail the workflow task on a determinism violation — a real, automatable check).
9. **Notification idempotence** — running `notify` twice with the same `(thread_id, kind)` leaves exactly one record.
10. **`portfolio.db` mtime unchanged** before/after the full suite — every phase runs this.
11. **Real-run gate (manual, per project convention)** — one real scheduled trigger against the dev server with real models and the real portfolio, watched in the Temporal UI, ending in a genuinely pending approval that Abhi then answers **by hand** through the Phase 8 path. Every LLM-touching phase so far had a bug only a real run found; assume this one does too.

No Playwright — the Temporal Web UI is third-party software, not this project's UI surface.

## Open Decisions (need Abhi's call before planning)

1. **The new `temporalio` dependency.** ~15 MB prebuilt `abi3` wheel, four small transitive deps, Apache-2.0, works on this venv's Python 3.14/win_amd64 with no compiler. Unavoidable for this phase — but it is a new dependency and the rule says surface it, not fold it in. **Nothing has been installed.**
2. **Notification channel** — `plan.md`'s explicit open item. Options, with the real setup cost of each: **(a) local-only** (store record + console + Temporal UI) — zero setup, zero secrets, requires Abhi to look; **(b) email via SMTP** — needs a Gmail app password in `.env`, and this project has already had three key-paste incidents; **(c) Telegram/Discord bot** — a bot token, genuinely good push-to-phone, another secret; **(d) ntfy.sh** — no account, push to phone, but the payload traverses a third-party host (a portfolio recommendation is not nothing); **(e) Windows toast** — local, no secret, but only while Abhi is at the machine. **Proposed: (a) now, with the notify Activity shaped so a channel is a drop-in later.** Explicitly **not** assumed: no email is sent from this design.
3. **Does a human-readable notification file get written?** A `sandbox-data/advisor-monitor.log` (JSONL) would be far easier to skim than a store row. But `CLAUDE.md` says agent writes go to `agent_memory.db` only, and quietly adding a second write surface is exactly the kind of drift that rule exists to stop. **Proposed: no file by default; ask.**
4. **If a prior approval is still pending, should the daily run skip?** **Proposed: yes** — it prevents unbounded pending threads piling up, and avoids paying daily for reports nobody has read. **The cost of saying yes:** one forgotten pending approval silently stops all future runs until it's answered. Mitigated by the "still pending" notification firing every day, but Abhi should decide whether that failure mode is acceptable versus letting runs accumulate.
5. **Does the monitor read `fired_alerts`/`daily_performance`?** `plan.md`'s Temporal sketch mentions it; no node in the graph reads them today. **Proposed: no** — a data-and-judgment feature, not a scheduling one, and folding it in here is silent scope growth. Named so it isn't lost.
6. **Cadence and time of day.** `plan.md` says daily; the time is unspecified. Options: after US close (~02:30 IST next day), after India close (~16:00 IST), or overnight. **This is also a cost decision** — daily is ~30 paid runs/month versus ~4 for weekly, on a long-term portfolio where the advice rarely changes day to day. **Proposed: daily at a fixed overnight IST time to start (it is what `plan.md` asks for and what exercises the durability learning goal), with the cost measured on the first real runs and weekly revisited if the output is stable.**
7. **Activity timeout numbers** — 45 min start-to-close / 2 min heartbeat are placeholders until a real full-graph run is timed. Flagged so they're set from measurement, not from a guess.

## Decisions (made autonomously, 2026-09-19 overnight run — Abhi explicitly delegated architectural calls for this session; each reversible, logged for morning review)

1. **`temporalio` dependency** — accepted. Small, permissively-licensed, prebuilt wheel already verified to work on this exact venv/OS with no compiler. Consistent with plan.md's stack; implement against it.
2. **Notification channel — (a), local-only, accepted as proposed.** No email/Telegram/Discord/ntfy — every alternative needs either a new credential (repeating this project's three prior key-paste incidents) or a third-party host seeing a real portfolio recommendation. Setting up any of those accounts/tokens is the same category of action as the LangSmith account gap from the observability phase: not something done on Abhi's behalf overnight. The notify Activity should be shaped so a real channel is a drop-in later, per the spec.
3. **No `advisor-monitor.log` file — declined, accepted as proposed.** Stick to the letter of "agent writes go to agent_memory.db only" rather than quietly widening the write surface; the Store record is the authoritative record, console output during an attended run is not a persisted write.
4. **Skip the daily run if a prior approval is pending — accepted as proposed (yes).** Matches "avoid is the default" and avoids paying to generate reports nobody has reviewed yet. The pending-approval notification firing daily is the mitigation for the "one forgotten approval halts the monitor" edge case.
5. **`fired_alerts`/`daily_performance` — declined, accepted as proposed (out of scope).** A data/judgment feature, not a scheduling one; would be silent scope growth into this phase.
6. **Cadence — daily, accepted as proposed, but the live Schedule is NOT activated tonight.** Build and test the Workflow/Activity/Schedule-creation code, but do not call whatever creates a real recurring Temporal Schedule against production `.env` credentials. The design's own risk section names the first scheduled run as "this project's first unattended LLM spend" and recommends one attended run first — that attended run is Abhi's to trigger and watch, not something to set loose while he's asleep. Implementation should leave schedule activation as a clearly-documented manual step (e.g. a `scripts/`-level command or a one-line `create_schedule()` call Abhi runs himself), not something that fires automatically.
7. **Activity timeouts** — accepted as proposed; implementation should time a real (TestModel-driven, free) full-graph run via `WorkflowEnvironment`'s time-skipping test server to set them from measurement rather than guessing further.

## Risks / known weaknesses

- **This phase cannot be implemented before Phase 8 lands.** `build_compiled_graph`, the `interrupt()`, and the `Command(resume` grep test are all Phase 8 artifacts. As of writing, Phase 8 is **in progress on `phase-8-synthesizer`** — `build_compiled_graph` and `human_approval_gate.py`'s `interrupt()` exist in that branch's working tree but are not merged or gated. Designing now is fine; building now is not, and the exact signatures above should be re-read from the merged code before planning.
- **The dev server needs one internet download and a hand-started worker.** Nothing here survives a machine reboot on its own — the worker and dev server must be restarted manually at P1. Durability within a run is real; durability of the *infrastructure* is a P2 deployment question.
- **The lock serialises correctly but does nothing about a genuinely stuck run.** A process that hangs holding the lock (a wedged MCP subprocess, say) blocks everything until it is killed. The Activity's heartbeat timeout catches this on the Temporal side; a manual run has no such watchdog.
- **Skipping on a prior pending approval (Open Decision 4) is a real stop-the-world lever.** One unanswered approval halts the monitor indefinitely. Deliberate and reported daily, but worth naming as the sharpest edge in the design.
- **`temporalio.contrib.langgraph` may fix the Store limitation later**, which would make per-node activity decomposition genuinely attractive (per-node retries and Temporal-UI visibility). The coarse-Activity design here is the correct call *today*, against the verified 1.33.0 behaviour — it is not a permanent judgment. Worth re-reading its README on any `temporalio` bump.
- **Two "did it run?" sources of truth.** Temporal's history and the checkpointer's threads can, in principle, disagree (an Activity that ran the graph and then failed to report). The deterministic `thread_id` makes them reconcilable by hand, and the idempotent notify limits the blast radius, but this is the seam where a genuinely confusing bug would live.
- **Windows asyncio subprocess behaviour inside a Temporal worker is unverified.** The graph spawns three MCP stdio subprocesses; Temporal's worker runs its own event loop. This should work (async activities run on the worker's loop, and `asyncio.run` uses the Proactor loop on Windows), but it is exactly the kind of assumption this project's convention says to verify live before writing plan code — not after.
- **The first real scheduled run is also the first unattended LLM spend in this project.** Every prior real-model run was watched by a human as it happened. The existing caps (3/group, 5/run per specialist) bound it, but "bounded" and "unwatched" together deserve one attended run before the schedule is left on.
