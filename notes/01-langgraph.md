# Learning Checkpoint — LangGraph

Written against `langgraph 1.2.11` / `langgraph-checkpoint-sqlite 3.1.1`, this project's actual graph (`src/agentic_frameworks/graph/graph.py`): `Research → Risk Analyst → {US Equity/ETF, India Equity, Mutual Fund, MF Holdings Analyzer} → Synthesizer → (human_approval_gate | END)`.

## 1. What problem does this solve that a plain LLM API call / plain cron job doesn't?

A single LLM call can't hold this system's actual shape: four independent research tracks that must each gather real data before any judgment happens, a point where they have to converge before a final answer can be produced, and a place where the whole run must *pause* — not error, not time out, genuinely pause — until a human answers. A plain script could fake the sequencing with function calls, but it can't durably pause mid-execution and resume days later from another process, and it can't give you a structural guarantee ("this node literally cannot proceed without the graph having a real checkpointer") the way a compiled graph with required keyword-only `checkpointer`/`store` arguments can (`build_compiled_graph`, Phase 8). A cron job runs and exits; it has no notion of "this run is 60% done, waiting on one more thing."

## 2. What did it make easy that was hard/verbose without it?

- **Fan-out/fan-in for free.** Four nodes reading the same upstream state and writing to different keys just needed `add_edge` from each to a common downstream node — LangGraph ran the Synthesizer exactly once, after all four completed, with no reducer, no manual `asyncio.gather` bookkeeping, no synchronization primitive I had to write. Verified live during Phase 8 design: this required *no* extra mechanism at all, which was itself a useful thing to confirm rather than assume.
- **Crash-safe pause/resume from a one-line primitive.** `interrupt()` inside `human_approval_gate_node` durably pauses a run — verified across a real second OS process reopening the same SQLite file and resuming it correctly (`test_gate_resume_survives_a_real_process_boundary`). Building this by hand would mean a whole state-serialization/resume-detection layer.
- **Resume re-runs only the interrupted node**, confirmed by call-counting (`synth` stayed at 1 call, `gate` went to 2, across a `Command(resume=...)`). This is what made "the approval gate must live downstream of the Synthesizer, never inside it" a correct design call rather than a guess — without this property, every approval would silently re-pay for the whole synthesis LLM call.

## 3. What did it make hard or awkward?

- **Version-specific footguns that only show up live.** Signature-based dependency injection (`store: BaseStore`, `config: RunnableConfig`) string-matches the *unevaluated* annotation when `from __future__ import annotations` is in effect. `Optional[BaseStore]` injects; `BaseStore | None` silently does not (Phase 7) — no error, just `store=None` arriving at a node that was compiled correctly. This is the kind of thing no amount of reading docs would have caught; it took an actual failing test to surface it, and it's exactly why this project's standing rule is "verify live, don't trust the documented API shape."
- **Interrupt without a checkpointer fails quietly.** `interrupt()` called on a graph compiled without a checkpointer doesn't raise — it just produces an unresumable pause. Both this project's store-using nodes and the approval-gate node needed an explicit, hand-written fail-loud guard (`if store is None: raise` / a checkpointer-presence check) because the framework won't do it for you.
- **`asearch`'s "prefix, not exact match" semantics** (Phase 7) are a real trap for anyone assuming namespace hierarchy gives isolation — a nested namespace search will happily return anything under it unless you filter for exact match yourself.

## 4. When would you reach for this in a real job?

Any workflow that is genuinely multi-step with real branching and a required human checkpoint — approval workflows, multi-stage document/data pipelines with a review gate, anything where "durably paused, resumable by a different process/person" is a real requirement rather than a nice-to-have. Not worth it for a single-shot "call the model once, get an answer" task — that's exactly the LLM-call case LangGraph is overkill for, and this project's own specialists still use plain `pydantic-ai` `Agent` calls, not a sub-graph, for that reason.
