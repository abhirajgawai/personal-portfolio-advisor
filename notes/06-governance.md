# Learning Checkpoint — Governance

Not a separate stack piece — plan.md names three mechanisms already built elsewhere in this project that together cover it.

## Policy enforcement

The New-Buy Conviction Gate (portfolio-fit → fundamentals → valuation → style-fit for equities; the 7-check equivalent for mutual funds) is a real reject-early policy layer, not advisory text a model is asked to follow. Tonight's build extended this in a specific way worth naming: guardrails in this project are structural (a field the LLM's schema physically cannot contain, a tool that isn't in the registered toolset, a hard numeric cap enforced by `Field(le=...)`) rather than instructional (a prompt asking the model to behave). The clearest example is Phase 8's human-approval gate — nothing in the codebase can construct a resume command (enforced by an AST-based test banning any reference to `Command` under `src/`), which is a materially stronger guarantee than "the prompt says never auto-approve."

## Audit trail

Every decision — specialist verdicts, rejected candidates (Phase 7's `RejectedCandidate` records), the Synthesizer's final report, pending/answered approvals — lands in `agent_memory.db`, never `portfolio.db`. This is real and already queryable, not aspirational: Phase 7's rejection-memory namespace and Phase 8's `AdvisorReport` are both durable rows in that one file, and the LangGraph checkpointer gives a full state history per thread on top of that.

## Cost/usage tracking

This is the one leg of the governance triad that is **not yet real** — it's meant to come from LangSmith tracing (per-node, per-run token cost and latency), and that phase is blocked on a LangSmith account only Abhi can create (see `notes/04-observability.md`). What exists today instead: hard, code-enforced caps on generation volume (3 candidates/sector/specialist, 5/run, `MAX_NEW_BUY_SUGGESTIONS`, `MAX_NEW_CAPITAL_PCT`) and Phase 7's rejection memory, which exists specifically to stop the same candidate being re-evaluated (and re-paid for) across runs. That's cost *control*, enforced structurally — but it isn't cost *visibility* in the sense LangSmith would provide, and the two aren't the same thing.

## Honest gap (plan.md names this explicitly)

There is no access-control or compliance layer — no notion of who is allowed to approve a recommendation, no audit of *who* clicked approve versus just that `approved_by` is a non-empty string, no regulatory sign-off concept. Reasonable to skip for a solo local tool used by one person, and this note exists specifically so that's a stated decision rather than an unexamined gap. If this system were ever used by more than one person, or connected to anything that could actually place a trade, this is the first thing that would need to become real rather than deferred.
