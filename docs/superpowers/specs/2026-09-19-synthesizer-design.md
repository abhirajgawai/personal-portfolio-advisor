# Recommendation Synthesizer + Graph Fan-In + Human-Approval Gate — Design Spec

**Status:** Draft for review (designed 2026-09-19; needs Abhi's sign-off on the Open Decisions — especially the model choice and the position-sizing numbers — before planning)
**Phase:** 8 of the Portfolio Advisor Fleet LangGraph build (see `plans/plan.md`)
**Depends on:** Phase 7 (`phase-7-langgraph-store` branch) — the `CandidateEvaluation` `identifier`/`identifier_kind`/`display_name` convention and `RejectedCandidate` are both load-bearing here.

## Goal

Close the graph. Today `Research → Risk Analyst → {4 parallel branches}` all dead-end at `END`, so the system produces four disconnected verdict lists and no actual advice. This phase adds the last two nodes `plan.md` calls for:

1. **`synthesizer_node`** — reconciles all four branches into one strictly-validated, genuinely two-sided report (existing-holdings guidance *and* new-buy suggestions), and
2. **`human_approval_gate_node`** — a real LangGraph `interrupt()` that suspends the run in a genuinely-pending state on **any** sell/reduce/add-to or new-buy suggestion, per the non-negotiable guardrail.

This is also the first node in the project whose job is *judgment across specialists* rather than judgment of a single instrument, and the first place the graph stops mid-run and waits for a person.

## Scope

**In scope:**

- `src/agentic_frameworks/graph/nodes/synthesizer.py` (new) — one LLM call per run, no toolsets.
- `src/agentic_frameworks/graph/nodes/human_approval_gate.py` (new) — no LLM, no tools, no I/O beyond `interrupt()`.
- `src/agentic_frameworks/graph/schemas.py` (modified) — `SynthesizedRecommendation` (the LLM's `output_type`), `AdvisorReport` + its sub-models (what lands in state), `HumanApproval` (the resume payload contract).
- `src/agentic_frameworks/graph/state.py` (modified) — `SynthesizerState`, `ApprovalState`; `GraphState` gains `recommendation` and `approval`.
- `src/agentic_frameworks/graph/graph.py` (modified) — the four `→ END` edges become `→ synthesizer`; a conditional edge from `synthesizer`; plus a new `build_compiled_graph(*, checkpointer, store)` factory (see Decision 5).
- `src/agentic_frameworks/graph/models.py` (modified) — `get_synthesizer_model()` with its own named constants, separate from `get_specialist_model()`.
- Tests per Testing below.

**Out of scope (explicitly deferred, with reasons):**

- **Per-holding judgment of the 36 IN equities, 3 US equities and 4 ETFs.** No node in this system has ever judged them — `mutual_fund_holdings_analyzer_node` covers the 9 funds only. The Synthesizer will **not** invent fundamentals-grade verdicts on 43 securities it has no fundamentals data for; that would be exactly the fabrication `CLAUDE.md`'s rule 5 forbids and would collapse the gather/judge separation (principle 4). Those holdings get **allocation-level** guidance grounded in real Risk Analyst measurements, and the coverage gap is reported *in the output* (see `ReportCoverage`), not hidden. An "Equity Holdings Analyzer" — the direct-equity mirror of the fund analyzer — is the obvious Phase 9 candidate and is named here rather than quietly skipped.
- **Real estate** stays a fixed input reasoned *around* (34% of the portfolio, illiquid, no price feed) — it appears in the allocation/liquidity narrative, never as a hold/sell/buy subject. Same rule every prior phase followed.
- **Writing the approval decision back to the store** as long-term memory ("Abhi rejected this suggestion") — a genuinely useful future cost control, but it's a new namespace, a new staleness question, and a new identity question on top of a phase that already adds two nodes. Deferred, not forgotten.
- **Any UI for the approval step.** The gate produces a pending interrupt; how a human sees and answers it (CLI prompt, Studio, Temporal notification) is the *caller's* job and belongs to the Studio/Temporal phases. This phase ships the graph-side contract and a test that drives it.
- **Studio/LangSmith, Temporal** — still their own phases.

## Background / Constraints (all live-verified against the installed versions, 2026-09-19)

Installed: `langgraph 1.2.11`, `langgraph-checkpoint 4.2.0`, `langgraph-checkpoint-sqlite 3.1.1`, `pydantic-ai 2.42.0`, `pydantic 2.13.5`.

- **Fan-in needs no new mechanism and no reducers.** A probe graph with the exact shape this project has — four parallel nodes off one parent, each writing a *different* `TypedDict` key, all four with a plain `add_edge(branch, "synth")` — ran `synth` **exactly once, after all four completed**, with all four keys visible in `state`. Execution order observed: `n_a, n_b, n_c, n_d, synth`. **Consequence:** plain `add_edge` is the fan-in mechanism. No `Annotated[..., operator.add]` reducer, no `Send`, no `defer=True`. Adding a reducer would be unnecessary *and* would change the semantics of state fields that four existing, passing nodes already rely on.
- **`interrupt()` suspends and resumes exactly as needed — verified end to end, including across processes.** `langgraph.types.interrupt(value)` (signature `(value: Any) -> Any`) inside a node returns `__interrupt__` in the `ainvoke` result, leaves `state.next == ("human_approval_gate",)` and `state.interrupts` populated. A **second, separate OS process** opened the same SQLite checkpoint file, saw the same pending `next`/interrupt, and resumed with `Command(resume={...})`, which returned the resume payload as `interrupt()`'s return value.
- **Resume re-runs only the interrupted node, not the upstream LLM node.** Counter-instrumented probe: after resume, `synth` call count stayed at **1** while `gate` went to **2**. In the cross-process probe, `synth`'s print never appeared in process 2 at all. **Consequence — this is the single most important structural decision in this design:** the `interrupt()` must live in its **own downstream node**, never inside the Synthesizer. Putting it in the Synthesizer would re-execute (and re-pay for) the entire synthesis LLM call on every approval.
- **`interrupt()` without a checkpointer does *not* raise — it silently produces an unresumable "pause".** Verified: compiling without `checkpointer=` and hitting `interrupt()` returned `{'report': ..., '__interrupt__': [Interrupt(...)]}` with no error and no way to resume. This is the same class of silent degradation as Phase 7's `store=None` hazard, and is handled the same way: a fail-loud guard (see Error Handling) plus a compile-time factory that makes the mistake unreachable.
- **A node can see the checkpointer by signature injection.** `async def node(state, config: RunnableConfig)` receives the config; `config["configurable"]["__pregel_checkpointer"]` is the live `AsyncSqliteSaver` when compiled with one and **`None`** when not. Verified both ways. `config` is injected under the *evaluated* type (unlike Phase 7's `store`, which needed the literal `Optional[BaseStore]` source-string form) and tests calling the node directly can pass a plain dict — so this guard does not break the direct-call node-test pattern every existing test uses.
- **`interrupt_before=` / `interrupt_after=` are rejected as the mechanism.** They are compile-time arguments (verified present in `StateGraph.compile`'s signature), which means any call site can simply omit them and get a graph with no approval gate. `interrupt()` inside node code cannot be compiled away. Principle 7: guardrails are structural, not configurational.
- **`pydantic-ai` has the two validation mechanics this design needs.** `Agent.output_validator` exists on 2.42.0 and `ModelRetry` is importable — together they give a *bounded* "the model named an identifier that doesn't exist, try again" loop that then fails loud, rather than an unbounded retry or a silent accept. `pydantic.Field(max_length=N)` on a list field raises `ValidationError`, which `pydantic-ai` surfaces as a retry — so hard caps can be expressed in the schema and re-asserted in code.
- **`TestModel` produces valid instances of a deeply nested output schema** (verified against a Synthesizer-shaped model with two nested lists) — but it emits the *first* `Literal` member, minimum-bound floats, and **exactly one element per list**. It therefore cannot exercise the empty-`new_buy_suggestions` path or a bad-identifier path; `FunctionModel` is required for those (same reasoning that forced Phase 7's deterministic fake evaluator).
- **Real platform names, read live from `portfolio.db` under `mode=ro` (mtime confirmed unchanged):** `zerodha`, `moomoo`, `groww`, `sharekhan`, plus a `real_estate` pseudo-account. `moomoo` is the only non-INR (SGD) account and is where every US holding sits. These are **derived at runtime** from `state["holdings"]`'s `platform` values, never hardcoded — same rule as live gap computation in Phases 4/5.

## Architecture

Five ideas, in order.

### 1. Fan-in is plain edges; the Synthesizer is a single-writer node

```
                         ┌─ us_equity_etf_specialist ──┐
                         ├─ india_equity_specialist ───┤
risk_analyst ─── fan-out ┤                             ├── synthesizer ─┬─(needs approval)→ human_approval_gate → END
                         ├─ mutual_fund_specialist ────┤                └─(nothing consequential)───────────────→ END
                         └─ mutual_fund_holdings_analyzer ┘
```

Each of the four branches already writes one disjoint `GraphState` key. The Synthesizer reads all four plus `holdings`/`allocation` (Research) and `concentration`/`category_overlap`/`currency_exposure`/`liquidity` (Risk Analyst), and writes exactly one new key, `recommendation`. The gate writes exactly one more, `approval`. No key is ever written by two nodes, so no reducer is needed anywhere in this graph.

**Reconciliation across branches is keyed off `identifier`, never list position** — the exact contract Phase 7's identifier fix established and `state.py`'s docstrings already mandate. `identifier_kind` tells the Synthesizer which namespace it is in: a `"scheme_code"` is not comparable to a `"ticker"`.

### 2. The Synthesizer judges; it does not gather, and it does not get new capabilities

**No toolsets.** Everything it needs is already in state, measured or judged by a node that had real data. Giving it `yfinance-mcp` would let it re-derive a fundamentals opinion that contradicts the specialist that actually ran the checks, and would make one node both gatherer and judge. It also means there is no tool surface here at all — the advisory-only guardrail is satisfied by absence, not by instruction.

**One LLM call per run.** Not one per holding. Cost control rule 4 (minimal context) is served by passing *derived summaries*, not raw state: the 45-row `holdings` dump never goes into the prompt; concentration is truncated to the top positions; non-Buy candidates appear as one line each, not with their full reasoning.

### 3. The LLM may never name a subject that the node did not give it

Phase 7's identifier fix split `ConvictionVerdict` (judgment, no identity) from `CandidateEvaluation` (judgment + code-bound identity) precisely so a model could not mislabel its own verdict. The Synthesizer is different — selecting *which* candidates to surface is genuinely its job, so it must be able to name them. The guardrail is therefore reshaped rather than dropped:

- The LLM's `output_type` (`SynthesizedRecommendation`) carries **`identifier` only** — no `identifier_kind`, no `display_name`. Those are re-bound in node code from a **subject index** the node built itself from the four branch outputs, exactly like `CandidateEvaluation.from_verdict`.
- An `@agent.output_validator` checks every `identifier` against that index, with the right *side* enforced too: an existing-holdings identifier must come from the existing-holdings set, a new-buy identifier from the **Buy-verdict** candidate set. An unknown or cross-contaminated identifier raises `ModelRetry` (bounded by `retries=2`), then fails loud.

Net effect: a hallucinated ticker cannot reach a recommendation, and a name can only come from the same verified source that produced the identifier.

### 4. "Avoid is the default" is preserved structurally, in four layers

1. **Code filters to `action == "Buy"` before the call.** Non-Buy candidates are never presented as buyable options. They appear only in a short "already considered and rejected this cycle" context block, and the output validator refuses their identifiers in `new_buy_suggestions`.
2. **The Synthesizer may only shrink the Buy set, never grow it.** Enforced by the validator's membership check — a subset is always legal, a superset is structurally impossible.
3. **Empty is schema-valid and needs no special case.** `new_buy_suggestions: list[NewBuySuggestion]` has no minimum length. A `model_validator` enforces the biconditional `new_buy_suggestions == [] ⟺ no_new_buy_reason is not None`, so "nothing this cycle" is a first-class, *explained* result rather than a silently empty field.
4. **When the Buy set is empty to begin with, the new-buy half is skipped entirely** — the prompt omits the new-buy block, and `no_new_buy_reason` is filled **deterministically in code** with the real counts (how many candidates each specialist proposed, evaluated, and rejected, and — if the store read is accepted, Decision 4 — how many were suppressed from memory). There is no reason to pay a model to write "no candidates cleared the gate," and a deterministic string can't drift.

### 5. The human-approval gate is a real interrupt, in its own node, that nothing in this system can answer

**Routing.** A pure, deterministic predicate over the finished report — not the model's opinion about whether approval is needed:

```
requires approval ⟺ new_buy_suggestions is non-empty
                  OR any existing_holdings_guidance.action in {"Sell", "Reduce", "AddTo"}
```

A Hold-only report with no new buys routes straight to `END` with no interrupt — verified that a conditional edge to `END` on this shape works. `AddTo` is included because it commits fresh capital; that inclusion is called out as Open Decision 2 rather than assumed.

**The gate node** is cheap by construction: no LLM, no tools, no store, no network. It builds a decision-request payload from the already-computed report, calls `interrupt(payload)`, validates whatever comes back against `HumanApproval`, and returns it. Because resume re-runs only this node (verified), an approval costs zero LLM tokens.

**Why nothing in this system can auto-approve it** — five independent reasons, four of them structural:

1. **Resumption is not reachable from inside the graph.** `interrupt()` returns a value supplied by `Command(resume=...)` passed to `ainvoke` by the **caller**, from outside the graph. No node can construct one for itself; there is no state field or edge that produces a resume value. Verified: in the cross-process probe the resume value came from the invocation, and the upstream node never re-ran.
2. **There is no default-approve path.** `HumanApproval` has no default for `decision` and no default for `approved_by` (`min_length=1`), `extra="forbid"`, and no timeout/auto-approve branch exists anywhere. A missing, empty, or malformed resume payload raises — it does not fall through to approved.
3. **A binding prohibition, pinned by a test, not a comment:** no module under `src/agentic_frameworks/` may construct `Command(resume=...)`. Testing item 9 greps the source tree for it. The Temporal phase inherits this: a scheduled run that finds a thread in pending-approval must leave it pending and notify — never resume it.
4. **`interrupt()` is in node code, not a compile flag**, so no call site can configure the gate away (this is why `interrupt_before=` was rejected).
5. **Defence in depth:** approval finalizes an advisory *report*. There is no order-placement tool anywhere in the fleet, so even a hypothetical auto-approve bug produces a marked-up document, not a trade.

## Components

### `schemas.py` (additions)

```
# --- what the LLM returns (identity-poor by design) ---
SynthesizedHoldingGuidance
    identifier: str                    # must be in the node's existing-holdings index
    action: Literal["Hold", "AddTo", "Reduce", "Sell"]
    rationale: str
    risk_flags: list[str]
    confidence: float (ge=0, le=1)

SynthesizedNewBuy
    identifier: str                    # must be in the node's Buy-candidate index
    rationale: str
    gap_addressed: str
    position_size_pct_of_portfolio: float (gt=0, le=MAX_NEW_POSITION_PCT)
    entry_approach: Literal["LumpSum", "Staggered"]
    suggested_platform: str            # validated against the live platform set in code
    risk_flags: list[str]
    confidence: float (ge=0, le=1)

SynthesizedRecommendation             # = the Agent's output_type
    portfolio_assessment: str
    allocation_notes: list[str] (max_length=ALLOCATION_NOTES_CAP)
    existing_holdings_guidance: list[SynthesizedHoldingGuidance] (max_length=...)
    new_buy_suggestions: list[SynthesizedNewBuy] (max_length=MAX_NEW_BUY_SUGGESTIONS)
    no_new_buy_reason: str | None
    # model_validator: (new_buy_suggestions == []) XNOR (no_new_buy_reason is not None)

# --- what lands in state (identity re-bound in code) ---
ExistingHoldingGuidance(SynthesizedHoldingGuidance)
    identifier_kind: Literal["ticker", "isin", "scheme_code"]
    display_name: str

NewBuySuggestion(SynthesizedNewBuy)
    identifier_kind: Literal["ticker", "scheme_code"]
    display_name: str
    specialist: Literal["us_equity", "india_equity", "mutual_fund"]

ReportCoverage                         # built in code, never by the LLM
    holdings_total: int
    holdings_individually_assessed: int
    unassessed_buckets: list[str]      # e.g. ["equity/IN", "equity/US", "etf/US"]
    coverage_note: str                 # why (no equity holdings analyzer exists yet)

AdvisorReport
    portfolio_assessment, allocation_notes, existing_holdings_guidance,
    new_buy_suggestions, no_new_buy_reason, coverage: ReportCoverage,
    generated_at: str                  # ISO-8601 UTC
    schema_version: int = 1
    @classmethod from_synthesis(synthesis, *, subject_index, coverage) -> AdvisorReport

# --- the resume contract ---
HumanApproval                          # model_config: extra="forbid"
    decision: Literal["approved", "rejected", "approved_with_exclusions"]
    approved_by: str (min_length=1)
    excluded_identifiers: list[str] = []
    note: str | None
    decided_at: str
```

Subclassing (rather than nesting) keeps the state dicts flat, matching the `ConvictionVerdict → CandidateEvaluation` precedent exactly.

### `state.py` (additions)

```
SynthesizerState(TypedDict): recommendation: dict   # AdvisorReport.model_dump()
ApprovalState(TypedDict):    approval: dict         # HumanApproval.model_dump()
GraphState: += recommendation: dict, approval: dict
```

### `models.py` (addition)

`get_synthesizer_model()` — its own named constants, mirroring `get_specialist_model()`'s `FallbackModel` pattern but with a **separate, explicitly chosen model list** (cost control rule 1: never a silent default, and never "whatever the specialists use" by inheritance). See Model Choice below.

### `nodes/synthesizer.py` (new)

Small private helpers, one job each — `_build_subject_index`, `_select_buy_candidates`, `_summarize_allocation`, `_summarize_risk`, `_summarize_considered_and_rejected`, `_compute_coverage`, `_build_synthesizer_agent`, plus the node.

### `nodes/human_approval_gate.py` (new)

`_requires_human_approval(report: dict) -> bool` (pure, tested standalone) and `human_approval_gate_node(state, config)`.

### `graph.py` (modified)

Four `add_edge(branch, END)` become `add_edge(branch, "synthesizer")`; `add_conditional_edges("synthesizer", _route_after_synthesis, ["human_approval_gate", END])`; `add_edge("human_approval_gate", END)`. Plus a new `build_compiled_graph(*, checkpointer, store)` with **both parameters keyword-only and required** — so there is no way to compile the advisor graph without the two things its guardrails depend on. `build_graph()` stays as-is (uncompiled) for Studio and for tests that want a different backing.

## Data Flow

**`synthesizer_node(state) -> SynthesizerState`**

1. **Build the subject index in code** (`{identifier: (identifier_kind, display_name, side, source)}`):
   - existing side ← `state["existing_fund_evaluations"]` (9 real funds, `isin`);
   - new-buy side ← `state["us_equity_etf_candidates"] + state["india_equity_candidates"] + state["mutual_fund_candidates"]`, **filtered in code to `action == "Buy"`**.
2. **Compute coverage** from `state["holdings"]` vs. the assessed set → `ReportCoverage`.
3. **Build minimal context** (cost rule 4): allocation totals + `byGeography`/`byAssetClass`; top-N concentration entries; `currency_exposure`; `liquidity`; the 9 fund verdicts verbatim (`action`, `reasoning`, `risk_flags`, `confidence`); the Buy candidates verbatim; the non-Buy candidates as one line each (`display_name` + `action` + first risk flag); the live platform set derived from `state["holdings"]`.
4. **Reject-early:** if the Buy set is empty, omit the new-buy block from the prompt entirely and pre-compute `no_new_buy_reason` deterministically.
5. **One `Agent.run`** — `output_type=SynthesizedRecommendation`, `retries=2`, **no toolsets**. The system prompt is a **byte-identical, stable rules block across runs** (cost rule 3 — prompt caching); *all* variable portfolio data goes in the user message. (This deliberately differs from the specialists, which bake `holdings_summary` into the system prompt and therefore can't actually cache it.)
6. **Validate + bind** via the output validator and `AdvisorReport.from_synthesis(...)`; `display_name`/`identifier_kind`/`specialist` come from the index, never the model.
7. `return {"recommendation": report.model_dump()}`

**`human_approval_gate_node(state, config) -> ApprovalState`**

1. `config["configurable"].get("__pregel_checkpointer") is None` → **raise** (an unresumable interrupt is a fake pause).
2. `interrupt({...})` with a compact decision request: the new-buy suggestions, the sell/reduce/add-to items, the assessment, and the coverage note. (Payload must be JSON-serializable for checkpointing — `model_dump()`, never a model instance.)
3. `HumanApproval.model_validate(resume_value)` — fails loud on anything malformed.
4. `return {"approval": approval.model_dump()}`

## Error Handling

Fail loud, no `try`/`except` swallowing — same standard as every prior node.

- **No checkpointer at the gate → `ValueError`.** The verified hazard: `interrupt()` without a checkpointer returns quietly and produces an unresumable run, which would look like "the approval gate ran" while being a no-op.
- **Unknown or cross-side `identifier` from the model → `ModelRetry` (bounded by `retries=2`), then propagates.** A recommendation naming a security nobody evaluated is the exact failure the identifier fix exists to prevent.
- **`suggested_platform` outside the live platform set → `ModelRetry`, then propagates.** Membership only; the geography-to-platform mapping stays prompt guidance (Open Decision 3).
- **Malformed resume payload → `ValidationError`.** No lenient parsing, no default-approve.
- **Missing upstream keys** (a branch that never ran) propagate as `KeyError` — a report synthesized from three of four branches is silently wrong, which is worse than a crash.
- **`portfolio.db` is untouched** by both new nodes; neither opens any database.

## Model Choice (cost control rule 1 — needs Abhi's explicit call)

The Synthesizer is the most judgment-heavy node in the graph, but it is **not doing primary analysis**: every fundamentals, valuation, NAV, benchmark and redundancy judgment has already been made by a specialist that had real tool data. Its actual job is reconciliation, prioritization, allocation-level narrative, and holding the "avoid is the default" line. That is a synthesis task, not a research task — which is why the flagship tier is hard to justify here.

Sizing: one call per run, roughly **8–12k input / 2–4k output tokens**.

| Option | OpenRouter price (per M in / out) | Est. cost/run | Note |
|---|---|---|---|
| `nvidia/nemotron-3-super-120b-a12b:free` | $0 / $0 | **$0** | 120B MoE reasoning model, 262k ctx, advertises `tools` + `structured_outputs` + `reasoning` |
| `deepseek/deepseek-v4-flash-0731:free` | $0 / $0 | **$0** | 1M ctx, structured outputs, same family already in use |
| `z-ai/glm-5.3-flash` (current specialist backup) | $0.09 / $0.30 | ~$0.002 | already proven in this project |
| `deepseek/deepseek-v4.1-flash` (current specialist primary) | $0.15 / $0.60 | ~$0.004 | proven, but "flash" tier — light reasoning |
| `nvidia/nemotron-3-super-120b-a12b` (paid) | $0.08 / $0.45 | ~$0.003 | the free primary's paid twin — no shared-pool 429s |
| `z-ai/glm-5.2` | $0.55 / $1.74 | ~$0.011 | genuine step-up |
| `z-ai/glm-5.3` (flagship) | $1.40 / $4.40 | ~$0.027 | ~$0.80/month once Temporal runs this daily |

**Proposed:** a three-model `FallbackModel` chain in `get_synthesizer_model()`:

1. `nvidia/nemotron-3-super-120b-a12b:free` — materially more reasoning depth than the flash tier the specialists use, at zero cost, which is exactly what tonight's constraint asks for;
2. `deepseek/deepseek-v4-flash-0731:free` — a second free leg, so one overloaded free pool doesn't push straight to paid;
3. `z-ai/glm-5.3-flash` — a cheap **paid** tail at ~$0.002/run, so a free-pool outage degrades the run instead of killing it.

This is the same `FallbackModel`-on-`ModelAPIError` pattern Phase 4 proved live after DeepSeek's free pool returned repeated 429s.

**The tradeoff being accepted, stated plainly:** free-tier models sit in a shared pool with real availability risk and less predictable instruction-following on a large structured schema than a flagship. The mitigations are the paid tail, `retries=2`, and — the real backstop — the fact that the identifier contract, the Buy-only filter, and the caps are all enforced in **code**, so a weaker model produces a worse *narrative*, never a wrong *subject* or an unvetted buy. If the real-model verification run shows it mangling cross-branch reconciliation, the step-up is a one-constant change and the cost is quantified above.

**Not assumed, must be verified in the real run:** that `structured_outputs` genuinely works for these two free models through `pydantic-ai`'s OpenRouter path rather than silently falling back to tool-mode.

## Testing (Phase 8 gate — must pass before the Studio/LangSmith phase is planned)

Real dependencies wherever free; `TestModel`/`FunctionModel` wherever a real call would cost money; `portfolio.db` never mocked; no fabricated tickers anywhere, including fixtures.

1. **Real state fixtures.** A session-scoped fixture that actually runs `research_node` then `risk_analyst_node` against the **real** `portfolio.db` (via `portfolio_data_mcp`) — so `holdings`/`allocation`/`holdings_by_bucket` and the four risk views are real shapes with real values. The four branch outputs are built from **real** identifiers already used elsewhere in the suite (real held-fund ISINs and names; real tickers such as `XLV`, `JPM`, `RELIANCE.NS`; real MFapi scheme codes such as `120605`/`118989`) — never invented ones.
2. **Pure helpers** — `_requires_human_approval` over hand-built reports (buy-only, sell-only, reduce-only, add-to-only, hold-only, empty); `_build_subject_index` over the real four-branch fixture; `_compute_coverage` returning the real assessed/total counts against the real portfolio.
3. **Schemas** — empty `new_buy_suggestions` with a `no_new_buy_reason` validates; empty with `None` fails; non-empty with a reason fails; `position_size_pct` bounds; `MAX_NEW_BUY_SUGGESTIONS` enforced by `max_length`; `SynthesizedRecommendation` provably has **no** `display_name`/`identifier_kind` fields (assert on `model_fields`) — the structural half of the identity guarantee; `from_synthesis` binds names from the index; `HumanApproval` rejects extra keys, a missing/empty `approved_by`, and an unknown `decision`.
4. **Identifier contract** — `FunctionModel` emitting an identifier not in the index causes retries then raises; an existing-holdings identifier used in `new_buy_suggestions` (cross-side contamination) likewise.
5. **Avoid-is-the-default** — a state where **every** candidate across all three specialists is `Hold`: assert `new_buy_suggestions == []`, `no_new_buy_reason` non-`None`, and — via a `FunctionModel` that captures the messages it receives — that the prompt contained **no** new-buy block at all.
6. **Minimal context** — the captured prompt does not contain the raw 45-row holdings dump, and does not contain non-Buy candidates' full `reasoning`.
7. **Fan-in, on the real graph** — compile the real `build_graph()` with a real temp-file checkpointer **and** store, real `portfolio.db`, `TestModel` patched into the five LLM nodes; assert via `astream` that `synthesizer` ran **exactly once**, after all four branches, and that its input state contained all four candidate keys.
8. **The approval gate is genuinely pending** — the same real graph, forced down the approval path: `ainvoke` returns `__interrupt__`; `aget_state().next == ("human_approval_gate",)`; `recommendation` is already persisted in state; then resume with `Command(resume=HumanApproval(...).model_dump())` and assert the synthesizer's call counter is **still 1** while the gate's is 2. Plus the **cross-process** variant (a second `subprocess` against the same checkpoint file) — that is what proves the pause survives the process, which is the whole point.
9. **Nothing can auto-approve** — a test that walks every `.py` under `src/agentic_frameworks/` and asserts none contains `Command(resume`. A structural prohibition enforced by CI, not by a comment.
10. **Fail-loud** — the gate raises when its config has no checkpointer; the no-approval-needed path (hold-only, no buys) reaches `END` with **no** `__interrupt__`.
11. **`portfolio.db` mtime unchanged** before/after the full suite — the guardrail check every phase runs.
12. **Real-model gate (manual, not automated, per project convention)** — one real run of the full graph against the real model pair, producing a real report from the real portfolio, with Abhi reviewing the output *and* exercising the approval interrupt by hand (approve one path, reject another). This is the only place real money is spent this phase — one Synthesizer call on top of the already-capped specialist spend. Phases 4, 5 and 6 each had a real bug that only a real run found; assume this one does too.

No Playwright — still no UI/browser surface.

## Open Decisions (need Abhi's call before planning)

1. **The model list** above (free-first three-model chain) — the one decision with a real cost/quality tradeoff. Alternatives and per-run costs are tabulated; nothing is silently defaulted.
2. **Does `AddTo` trigger the approval gate?** Proposed **yes** — it commits fresh capital, which is the thing the guardrail is protecting. `plan.md`'s wording ("any sell *and* any new-buy suggestion") is literally silent on adding to an existing position. Saying yes means more interrupts; saying no means a "double your RELIANCE.NS position" recommendation finalizes unreviewed.
3. **Position-sizing numbers.** `plan.md` lists this as an open item and the Synthesizer is the first node to need it. Proposed: `MAX_NEW_POSITION_PCT = 2.0` (per suggestion, as % of total portfolio) and `MAX_NEW_CAPITAL_PCT = 5.0` (sum across a run), both named constants, the per-suggestion one enforced in the schema and the aggregate in code. **The bound is the structural guardrail; the numbers are Abhi's call** — 2% of ₹10.29M is ~₹206k per suggestion, which is a real amount of money and should not be picked by an agent overnight.
4. **Should the Synthesizer read rejection memory from the store?** Phase 7's spec explicitly names this as the Synthesizer's job ("we skipped X because we rejected it in March") and shaped `RejectedCandidate` to make it possible without a migration. Proposed **in, read-only** — it makes the report honest about what was *not* considered and costs ~20 short strings. It does mean a second store-injected node and a second `store is None` guard.
5. **`build_compiled_graph(*, checkpointer, store)` as a new public factory in `graph.py`.** This changes `graph.py`'s current contract (it deliberately returns an *uncompiled* graph so call sites own the backing). Proposed: **add** it alongside `build_graph()` rather than replace — the compile-time requirement is what makes "you cannot get an unresumable approval gate" structural rather than a convention, but Studio will still want the uncompiled builder.
6. **Does a `rejected` approval decision end the run, or loop back?** Proposed: **end**. The report is persisted with the rejection recorded; re-running the graph is cheap and honest, whereas a revise-loop is an unbounded LLM cycle and this project has no evidence yet about what a good revision prompt looks like. Flagged because `plan.md` mentions cycles as a LangGraph learning goal, so Abhi may want the loop for that reason alone — in which case it needs its own iteration cap.

## Decisions (made autonomously, 2026-09-19 overnight run — Abhi explicitly delegated architectural calls for this session; each reversible, logged for morning review)

1. **Model list** — accepted as proposed (free-first `FallbackModel` chain: Nemotron-3-Super free → DeepSeek V4 Flash free → GLM-5.3-Flash paid tail). Zero primary cost, well-reasoned (Synthesizer reconciles already-tool-grounded specialist judgments rather than doing primary analysis), and the paid tail is already proven safe in this project from Phase 4.
2. **`AddTo` triggers the approval gate** — accepted as proposed (yes). It commits fresh capital, which is exactly what the guardrail exists to catch; erring toward more interrupts rather than fewer matches "avoid is the default."
3. **Position-sizing numbers — accepted as a DEFAULT, explicitly flagged as NOT a final decision.** Implementing with `MAX_NEW_POSITION_PCT = 2.0` / `MAX_NEW_CAPITAL_PCT = 5.0` as named, easily-changed constants, because the *structural* cap (a hard numeric ceiling enforced in code, never soft LLM guidance) is itself a non-negotiable rule from CLAUDE.md's cost-control section, and shipping the gate with a conservative, industry-common default is safer than shipping no gate at all for the rest of tonight's work. But the specific percentages are real-money parameters, not an engineering tradeoff — I'm treating this the same way I treated the LangSmith account gap: something within my authority to unblock the pipeline, but not to finalize. **Flagged in HANDOFF.md as needing Abhi's explicit sign-off before any suggestion sized against these constants is acted on** — the advisory-only guardrail means nothing gets executed either way, but the numbers should be reviewed before they're trusted, not just before they're traded on.
4. **Synthesizer reads rejection memory read-only** — accepted as proposed. Matches Phase 7's own stated purpose for the store, low cost, makes the report honest about what wasn't reconsidered.
5. **`build_compiled_graph(*, checkpointer, store)` as an additive factory** — accepted as proposed. Keeps `build_graph()`'s existing uncompiled contract for Studio while making the checkpointer/store requirement structural at the one call site meant to produce a real, resumable graph.
6. **`rejected` approval decision ends the run** — accepted as proposed. No revise-loop built; flagged in HANDOFF.md as a future decision if Abhi wants a bounded revision cycle later, matching `plan.md`'s cycles-as-learning-goal note.

## Risks / known weaknesses

- **Existing-holdings coverage is genuinely partial: 9 of ~49 holdings have a per-instrument verdict.** The two-sided guardrail is met (real verdicts on the funds, plus allocation-level guidance grounded in real Risk Analyst measurements), and `ReportCoverage` surfaces the gap *in the product*. But the honest statement is that "what to do with your 36 IN equities" is currently answered at the allocation level only. An Equity Holdings Analyzer is the natural Phase 9.
- **This is the first node whose output is pure opinion.** Every prior node either measured (Risk Analyst) or judged one instrument against real fetched data (specialists). The Synthesizer's `portfolio_assessment` and `allocation_notes` are free-text narrative with no tool grounding — validated in *shape*, not in *truth*. Mitigation is that every consequential field (identifier, action target, size bound, platform) is code-validated; the ungrounded part is prose. Worth re-reading critically in the real-model run.
- **The routing predicate is only as good as the report's `action` field.** If a model writes `action="Hold"` while the `rationale` argues for selling, the gate is bypassed. Not fixable by prompting; the mitigation is that `action` is a `Literal` the model must commit to, and the run is inspectable in Studio.
- **Free-tier availability.** Phase 4 was bitten by exactly this. The three-model chain with a paid tail mitigates it, but the first real run should be treated as a live test of the free tier, not just of the prompt.
- **A pending approval is an open thread with no owner yet.** Nothing in this phase notifies anyone that a run is waiting. Until the Studio/Temporal phases, that means a run can sit pending indefinitely. Acceptable now (runs are manual), and it must be designed for in the Temporal phase — which is also where the "never auto-resume" prohibition becomes load-bearing rather than theoretical.
- **SQLite concurrency remains single-process-verified only.** Phase 7 flagged this; the approval gate makes it slightly sharper, since a pending thread now has a genuinely long lifetime during which a second process could plausibly start another run.
- **Prompt-cache benefit is unproven on OpenRouter free models.** The stable-system-prompt design is correct regardless (cost rule 3), but whether these specific providers actually honor caching is not something this project has verified.
