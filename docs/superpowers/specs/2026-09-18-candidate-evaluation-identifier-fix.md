# `CandidateEvaluation` Identifier Fix — Design Note

**Status:** Implemented (not yet committed)
**Scope:** Bounded fix, not a phase — small enough to implement directly rather than go through the full spec → plan → SDD pipeline.
**Fixes:** the "Known, disclosed gap" carried since Phase 4 and re-flagged at the end of Phase 6 in `HANDOFF.md`.

## The bug

`CandidateEvaluation` — the structured output shared by all four specialist/analyzer nodes (`us_equity_etf_specialist`, `india_equity_specialist`, `mutual_fund_specialist`, `mutual_fund_holdings_analyzer`) — had **no field identifying which candidate or holding a verdict belongs to**. The subject of a verdict was knowable only two ways, both fragile:

1. **List position** — `existing_fund_evaluations[3]` is "whatever the 4th real mutual fund holding happened to be this run". Any reordering, filtering, dropped candidate, or partial failure silently re-labels every verdict after it.
2. **Free-text `reasoning`** — an LLM-written sentence that *usually* names the instrument. Not parseable, not validated, and not guaranteed to name the right one.

This mattered now rather than later because the Recommendation Synthesizer will consume all four lists at once, merge them, and route sell/new-buy suggestions through human approval. A mislabeled verdict at that point is a recommendation to buy or sell the wrong security — the most consequential failure mode this system has, and the one the guardrails exist to prevent.

## Decision 1 — one `identifier` + `identifier_kind`, not three optional typed fields

The three nodes' natural identifiers live in different namespaces, and each is already resolved and *verified* at the point the evaluation is built:

| Node | Identifier available | Where it comes from |
|---|---|---|
| US Equity/ETF Specialist | ticker (`AAPL`) | the symbol `_verify_ticker` confirmed against yfinance-mcp |
| India Equity Specialist | ticker (`RELIANCE.NS`) | same verification path |
| MF Holdings Analyzer | ISIN (`INF179K01WN9`) | `portfolio.db`'s `ticker` column for IN mutual funds — the real ISIN |
| MF Specialist (new-buy) | MFapi scheme code (`118632`) | `_resolve_scheme_by_name_and_direct_plan`, after the Phase 6 category-match check |

**Chosen:** a single required `identifier: str` plus a required `identifier_kind: Literal["ticker", "isin", "scheme_code"]`.

**Rejected:** three mutually-exclusive optional fields (`ticker: str | None`, `isin: str | None`, `scheme_code: str | None`). Pydantic cannot express "exactly one of these is set" without a custom validator, so the invariant would be a convention rather than a guarantee, and every consumer — starting with the Synthesizer — would have to write `ticker or isin or scheme_code` and decide for itself what to do when none or two are present. With one required field, "there is always exactly one identifier, and it is always present" is enforced by the type system at construction time.

`identifier_kind` is stored explicitly rather than inferred from the string's shape (`INF…` → ISIN, all-digits → scheme code, `.NS` suffix → NSE ticker). Shape-sniffing is a guess; this project does not guess at identity (fail loud, never fabricate).

## Decision 2 — `display_name` as well, sourced from the same verification step

`identifier` alone half-fixes the bug for mutual funds: `"118632"` is unusable in a recommendation shown to a human, and re-looking-up the name in the Synthesizer would be a second network call and a second chance to attach the wrong name. So `display_name: str` is required too, and always comes from the **same verified source as `identifier`**:

- equity/ETF: yfinance's `longName` for the symbol that was just verified;
- existing funds: `portfolio.db`'s real holding name;
- new-buy funds: the resolved MFapi `scheme_name` (already the Phase 6 "use verified identity downstream" fix).

This required one small, deliberate change: `_verify_ticker` now returns `str | None` (the verified `longName`, or `None` if the symbol doesn't resolve) instead of `bool`. The reject-early semantics are unchanged — `None` still means "drop this candidate" — but the verification call now yields the verified name it was already fetching and throwing away, so no extra call is needed and the name can only come from verification.

## Decision 3 — split the schema so the LLM *cannot* supply identity

`CandidateEvaluation` was the `output_type` of every Conviction Gate evaluator Agent. Simply adding `identifier` to it would have asked the model to echo back which candidate it was judging — exactly the class of mistake being fixed, and contradicting Phase 6's "use verified identity downstream" correction.

Instead the schema is split:

- **`ConvictionVerdict`** — the judgment only (`action`, `confidence`, `reasoning`, `risk_flags`, `entry_approach`, `suggested_platform`). This is now every evaluator Agent's `output_type`.
- **`CandidateEvaluation(ConvictionVerdict)`** — the verdict *plus* `identifier`, `identifier_kind`, `display_name`. Built in node code via `CandidateEvaluation.from_verdict(verdict, identifier=…, identifier_kind=…, display_name=…)`, and the only thing that ever lands in graph state.

This is CLAUDE.md's agentic design principle 7 (*guardrails are structural, not instructional*) applied literally: the model is never given the ability to name its own subject, so there is no prompt-adherence question about whether it names it correctly. Subclassing keeps the state dict **flat** (`{"identifier": …, "action": …, …}`) rather than nested (`{"identity": …, "verdict": …}`), which keeps the Synthesizer's per-candidate context small and readable and avoids re-shaping the four existing state fields.

## What changed

- `graph/schemas.py` — added `ConvictionVerdict`; `CandidateEvaluation` now subclasses it and adds the three identity fields plus the `from_verdict` classmethod.
- `graph/nodes/us_equity_etf_specialist.py` — `_verify_ticker` returns the verified name or `None`; evaluator `output_type` is `ConvictionVerdict`; the node binds `identifier=candidate.ticker`, `identifier_kind="ticker"`, `display_name=<verified longName>`.
- `graph/nodes/india_equity_specialist.py` — same (it imports `_verify_ticker` from the US node).
- `graph/nodes/mutual_fund_holdings_analyzer.py` — binds the holding's real ISIN and `portfolio.db` name, `identifier_kind="isin"`.
- `graph/nodes/mutual_fund_specialist.py` — binds `str(scheme_code)` and the resolved scheme name, `identifier_kind="scheme_code"`.
- `graph/state.py` — docstrings for `SpecialistState` / `GraphState` now state that consumers key off `identifier`, never off list position.
- `tests/graph/` — updated and extended; see below.

## Testing

Per project convention: `TestModel` for every LLM call, real dependencies everywhere they are free (`portfolio.db` via `portfolio_data_mcp`, `yfinance-mcp`, `mfapi_mcp` against live MFapi.in).

New/changed tests:

- `test_schemas.py` — identity fields are required (a verdict with no subject fails validation); `identifier_kind` rejects anything outside the three real namespaces; `ConvictionVerdict` provably has no identity fields; `from_verdict` binds identity to judgment and produces exactly the nine expected keys.
- `test_us_equity_etf_specialist.py` / `test_india_equity_specialist.py` — `_verify_ticker` returns a real name for `AAPL` (live yfinance) and `None` for a fake symbol; node output carries `identifier_kind == "ticker"` with non-empty identifier/name; and a dedicated test substitutes the verification step to prove `display_name` comes from verification rather than from the model.
- `test_mutual_fund_holdings_analyzer.py` — the 9 evaluations' `{identifier: display_name}` map equals the real 9 IN mutual fund holdings' `{ISIN: name}` map exactly, with `identifier_kind == "isin"` — an order-independent assertion, which is the whole point of the fix.
- `test_mutual_fund_specialist.py` — a surviving candidate carries the resolved scheme code and real scheme name. The resolve step is substituted with the live-verified real triple for scheme 118632 (`Nippon India Large Cap Fund - Direct Plan - Growth Option`, `Equity Scheme - Large Cap Fund`) because `TestModel` proposes a dummy name no real search can match; everything downstream (NAV history, metrics, benchmark) is real live data for that real scheme.

Full suite: **105 passed** (98 before this change, 7 new), no failures, no skips.

## Follow-ups / risks

- **Synthesizer must key off `identifier`, never list position.** That is now documented in `state.py`'s docstrings, and `identifier_kind` tells it which namespace it is looking at (a `"scheme_code"` identifier is not comparable to a `"ticker"` one).
- **The LangGraph Store spec (`2026-09-18-langgraph-store-design.md`) says "`CandidateEvaluation` unchanged" — no longer true.** No code conflict (that phase is unimplemented), but its `RejectedCandidate` schema should adopt the same `identifier`/`identifier_kind` convention so a rejection memo and an evaluation refer to the same thing the same way. Worth reconciling before that phase is planned.
- **No real-model run was performed for this change** — it is a schema/plumbing fix with no new prompt content and no new model behavior (the evaluator prompts are byte-identical; only the output schema shrank). The next real-model run of any specialist will exercise it incidentally, and per project rule that run still gates any phase that includes LLM behavior changes.
- **Not committed** — left in the working tree for the cross-checking process to review.
