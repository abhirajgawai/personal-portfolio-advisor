---
name: conviction-gate
description: Use before surfacing any new-buy candidate (stock, ETF, or mutual fund) in the portfolio advisor fleet. Applies the New-Buy Conviction Gate — reject early, don't research everything.
---

# New-Buy Conviction Gate

Applies to every new-buy specialist (US Equity/ETF, India Equity, Mutual Fund) in the portfolio advisor. The default outcome is **no candidate this cycle** — a suggestion is the exception, not the norm. Run the checks in order and reject at the first failure; don't do expensive research (fundamentals, valuation) on a candidate that fails the portfolio-fit check.

## The four checks, in order

1. **Portfolio-fit check** — does this candidate address a real, identified gap in the actual current holdings (queried from `instrument`/`position` in `portfolio.db`, not assumed)? No identified gap → reject immediately, before any further research.
2. **Fundamentals check** — pull real financials via the research MCP tool: revenue/earnings trend, debt levels, margins. Never reason from model priors alone.
3. **Valuation check** — is it reasonably priced now vs. its own history and peers? Given current market conditions, richly-valued candidates are flagged or rejected — never rationalized into a "buy anyway."
4. **Style-fit check** — is it consistent with the cap-size/sector tilt the actual holdings show (derived from a real query, not a guess)? A deviation is allowed only with explicit justification in the output.

## Output contract

Every run of the gate produces one of:
- **A candidate that cleared all four checks**, with the finding from each check attached as part of the reasoning (not just a final verdict).
- **A rejection**, naming which check failed and why. This is a normal, expected, and equally valuable result — do not treat "found nothing to suggest" as a failure of the run.

Never skip a check to produce a result faster. Never surface a candidate that hasn't cleared all four.
