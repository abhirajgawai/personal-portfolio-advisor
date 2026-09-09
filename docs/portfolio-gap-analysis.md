# Portfolio Sector Gap Analysis

**Status:** Findings from a spike (not a design doc) — queried live from `portfolio.db`, read-only, on 2026-09-09.
**Purpose:** Gives the future new-buy specialists' portfolio-fit check (first stage of the Conviction Gate) real gaps to check candidates against, instead of guessing. Per `plans/plan.md`'s Open Items.

## Method

Queried `position`/`instrument` directly (read-only) for `asset_class='equity'`, grouped by `sector` and `geography`. Compared against GICS's 11 standard sectors as a reference taxonomy — not because this project adopts GICS formally, but because it's a recognizable, complete baseline to spot what's missing.

## Data-quality note (not a gap, a caveat)

Two IN equity rows have no `sector`: `MANUAL_TOTAL` entries for the Groww (₹1,02,730) and Sharekhan (₹37,868) platforms — these are manually-tracked lump-sum totals, not individual stock positions, so they carry no per-security sector data. They're excluded from the sector breakdown below; don't read their absence as "no sector assigned" in the data-quality sense — there's nothing to assign it to.

## IN Direct Equity — ₹18.78L total, by sector

| Sector | Value (₹) | % of IN equity | GICS-ish bucket |
|---|---:|---:|---|
| Industrial Equipment | 3,79,764 | 20.2% | Industrials |
| Financial Services | 2,95,216 | 15.7% | Financials |
| Pharmaceuticals | 2,42,904 | 12.9% | Health Care |
| Industrials | 1,34,426 | 7.2% | Industrials |
| Steel | 1,10,048 | 5.9% | Materials |
| Water & Environment | 96,403 | 5.1% | Utilities |
| Consumer Packaging | 82,873 | 4.4% | Consumer Staples |
| Renewables | 64,554 | 3.4% | Utilities |
| Power | 48,805 | 2.6% | Utilities |
| Auto Components | 48,126 | 2.6% | Consumer Discretionary |
| Information Technology | 47,101 | 2.5% | Information Technology |
| Banking | 41,841 | 2.2% | Financials |
| Agri & Sugar | 30,498 | 1.6% | Consumer Staples |
| Chemicals | 28,218 | 1.5% | Materials |
| Electrical Equipment | 27,512 | 1.5% | Industrials |
| Infrastructure | 25,397 | 1.4% | Industrials |
| Media & Entertainment | 24,001 | 1.3% | Communication Services |
| Travel & Tourism | 9,939 | 0.5% | Consumer Discretionary |

**Rolled up by GICS-ish bucket:** Industrials (32.9%), Financials (17.9%), Health Care (12.9%), Utilities (11.1%), Materials (7.4%), Consumer Staples (6.0%), Consumer Discretionary (3.1%), Information Technology (2.5%), Communication Services (1.3%).

**Genuinely absent (0%) from IN equity:** Energy, Real Estate (as an *equity* sector — separate from the direct real-estate holding, which is a different asset class entirely), and any real Consumer Discretionary depth beyond a small auto-components + travel sliver.

**Thin relative to India's market weight:** Information Technology at 2.5% is notably underweight — IT is one of the largest sectors on Indian exchanges (TCS, Infosys, HCL Tech, etc.), and this portfolio holds almost none of it directly. Consumer Discretionary (3.1%) and Communication Services (1.3%, telecom specifically absent) are similarly thin.

## US Equity + ETF — by sector

| Sector | Asset class | Value (₹) | Note |
|---|---|---:|---|
| Semiconductors | ETF (SMH) | 7,12,689 | |
| US Large Cap | ETF (VOO) | 6,60,932 | Broad index — not a sector bet |
| Technology | Equity (GOOGL, IBM) | 5,10,101 | |
| Gold | ETF (IAU) | 2,01,210 | Commodity, not equity sector |
| Semiconductors | Equity | 94,964 | |

**Genuinely absent from US holdings:** every GICS sector except Technology/Semiconductors and the broad-market VOO index — no Health Care, Financials, Energy, Industrials, Consumer (Discretionary or Staples), Real Estate, Utilities, Communication Services, or Materials exposure at all, direct or via a sector ETF. VOO gives diversified exposure in aggregate, but it's the only non-tech line in the US book.

## Summary — real gaps for the specialists to check against

- **India:** Energy and equity-sector Real Estate are zero. IT, Consumer Discretionary, and Communication Services are thin relative to their actual weight in the Indian market.
- **US:** Everything outside Technology/Semiconductors and the VOO index is zero — Health Care, Financials, Energy, Industrials, Consumer, Real Estate, Utilities, Communication Services, Materials.
- **Cross-check against Risk Analyst output:** this rolls up cleanly with the category-overlap risk view already built — "Small Cap" showed up as a cross-asset-class overlap; sector concentration in IN Industrials (32.9%) is a similar signal, now visible at the sector-taxonomy level this analysis adds.

This list is descriptive, not prescriptive — it says what's *absent*, not what to buy. Whether any of these gaps are worth filling is exactly what the future specialists' portfolio-fit check (plus fundamentals, valuation, and style-fit) decides, one real candidate at a time.
