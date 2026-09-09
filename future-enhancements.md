# Future Enhancements

Deferred ideas surfaced during design/brainstorming sessions — not part of the current phase's scope, tracked here so they aren't lost.

## Risk Analyst node (Phase 2)

- **True statistical correlation** — pairwise correlation coefficients computed from historical price/return time series per holding (via `holding_history` or similar), instead of the category-overlap heuristic used in the first pass. Deferred because it needs new data-fetching work and clean price history isn't guaranteed for mutual funds/real estate.
- **Horizon-fit risk check** — whether concentration/illiquidity/equity-heaviness make sense given the investor's actual time horizon. Deferred because it needs a stated time horizon input that doesn't exist yet.
- **Drawdown context** — how holdings have historically behaved in past downturns, to contextualize concentration risk. Deferred because it needs historical price data the Research node doesn't currently fetch.
- **Allocation drift detection** — comparing current allocation against a *target* allocation. Deferred until a target allocation is defined; until then the node can only report current allocation, not drift.
- **Fund-level lock-in nuance for liquidity risk** — liquidity in the first pass is a simple asset-class split (`real_estate` = illiquid, everything else = liquid). This misses that some mutual funds (e.g. ELSS) have a lock-in period even though they're not "illiquid" in the real-estate sense. Deferred until it's confirmed whether `portfolio_data_mcp` carries fund-type/lock-in data at all.
