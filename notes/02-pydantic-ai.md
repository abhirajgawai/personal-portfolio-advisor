# Learning Checkpoint — Pydantic AI

Used in every LLM-calling node: all 3 new-buy specialists, the MF holdings analyzer, and the Synthesizer, all via `pydantic_ai.Agent` with a typed `output_type`.

## 1. What problem does this solve that a plain LLM API call / plain cron job doesn't?

"Just parse the model's JSON" breaks the moment the model wraps output in prose, uses a slightly different key name, or returns a value out of range (this project hit exactly that in Phase 4: a real run once returned `confidence=7.0` on a field with no declared bound — caught only because the real-model gate is a required step, not because any amount of `TestModel`-only testing could have found it). Pydantic AI turns "the model claims this is the answer" into "the model produced something a schema will actually validate, and if not, it gets a bounded number of chances to fix it" (`Agent(retries=2)`, `ModelRetry`).

## 2. What did it make easy that was hard/verbose without it?

- **Structural guardrails, not instructional ones.** The single most-repeated pattern across every phase this session: split the LLM's `output_type` so it *cannot* contain a field it shouldn't be trusted with, then bind that field in code from already-verified data. `ConvictionVerdict` (judgment only) → `CandidateEvaluation` (verdict + code-bound identity) in Phase 6/7; `SynthesizedRecommendation` (identifier only) → `AdvisorReport` (+ code-bound `display_name`/`identifier_kind`) in Phase 8. This only works cleanly because Pydantic AI's `output_type` is just a Pydantic model — subclassing and building the richer object in node code is trivial.
- **`output_validator` as a second, semantic gate beyond field-level validation.** Type/range checks alone can't verify "this identifier is actually in the set of things we asked about" or "the total of these five percentages doesn't exceed a portfolio-level cap" — those needed a real function checking the *whole* output against external context (a subject index built from state), which `output_validator` supports directly, raising `ModelRetry` to give the model one bounded chance to correct itself before failing loud.

## 3. What did it make hard or awkward?

- **`TestModel`'s determinism is a testing limitation, not just a convenience.** It always emits exactly one list element and the first `Literal` value by default — which means it structurally cannot exercise an empty-list path, a specific rejected-verdict path, or any negative case. Every phase that needed to test "the model said Hold" or "the model proposed nothing" had to drop to `FunctionModel` (a hand-written fake response function) instead — this was worth knowing before writing the first Phase-7 suppression test, which initially and silently never exercised its own "Hold" branch because of exactly this.
- **No visibility into whether a given OpenRouter model actually honors structured outputs** until you call it for real. The free-tier models chosen for the Synthesizer (`nvidia/nemotron-3-super-120b-a12b:free`, `deepseek/deepseek-v4-flash-0731:free`) are unverified on this specific point — flagged explicitly as something only the real-model gate can confirm, not something inferable from a model card.

## 4. When would you reach for this in a real job?

Any place an LLM's output feeds a downstream system automatically (not just displayed to a human who can sanity-check it) — anywhere a malformed or out-of-range value would silently corrupt data, trigger a wrong branch, or (as in this project) attach real financial data to the wrong instrument. Less necessary for a pure chat/summarization feature where a human reads the raw text output directly.
