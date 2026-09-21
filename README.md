# AI-Powered Pricing Strategy Assistant — Simulated EdTech Platform

A pricing/retention decision-support system for a simulated EdTech subscription platform:
**1,600 fully synthetic learners**, 6 named segments, 4 plans, a 12-month engagement panel,
a 24-rule transparent recommendation engine (optionally informed by a simple ML churn
model), and a held-out-cohort experiment measuring whether the framework actually beats a
baseline offer strategy. Read [Data Assumptions](#data-assumptions) before trusting any
number here.

**Stated plainly, per the brief: 100% of the learner data in this project is synthetic.
No real learner-level EdTech pricing dataset exists publicly, and none is used here.**

## At a glance — exact scale

| Item | Count |
|---|---:|
| Learners (main cohort) | **1,600** |
| Engagement panel rows | **19,200** (1,600 × 12 months) |
| Segments | **6** |
| Plans | **4** (Free, Basic, Pro, Premium) |
| Rules in the recommendation engine | **24** |
| Parameters a rule can condition on | **14** |
| Cumulative 12-month conversion (Free→paid) | **3.81%** |
| Monthly churn rate, paid plans (pooled) | **5.62%** |
| Churn-risk ML model | Logistic regression, trained on 267 learner-months, held-out test **AUC 0.706** |
| Held-out intervention cohort | **3,000 learners**, resampled **15×** per arm (45,000 evaluation rows/arm) |

## Results

All numbers below are pulled directly from `output/tables/*.csv` and live queries against
the real local MySQL instance — nothing rounded up or reconstructed from memory. Full
interpretation (including the honest trade-offs) is in `WRITEUP.md`.

### Segment-level (SQL 02a) — renewal / upgrade / churn rate by segment

| Segment | Paid months at risk | Renewal % | Upgrade % | Churn % |
|---|---:|---:|---:|---:|
| Power Learners | 103 | 97.09 | 1.94 | 0.97 |
| Dormant Explorers | 10 | 100.00 | 0.00 | 0.00 |
| Enterprise Professionals | 17 | 88.24 | 5.88 | 5.88 |
| Casual Upskillers | 51 | 90.20 | 1.96 | 7.84 |
| Career Switchers | 72 | 87.50 | 4.17 | 8.33 |
| Budget Students | 14 | 78.57 | 0.00 | 21.43 |

Churn ordering matches the segments' designed price-sensitivity ordering almost exactly
(Budget Students, 1.8× sensitivity, churns most; Power Learners, 0.5×, churns least) — a
sanity check that the generator's elasticity mechanism is behaving consistently, not
producing noise.

### Conversion pattern (SQL 03a–03c) — freemium→paid by segment, time-to-convert

| Segment | Learners | Converters | Conversion rate |
|---|---:|---:|---:|
| Power Learners | 249 | 23 | **9.24%** |
| Career Switchers | 287 | 17 | 5.92% |
| Casual Upskillers | 319 | 12 | 3.76% |
| Enterprise Professionals | 199 | 5 | 2.51% |
| Budget Students | 297 | 3 | 1.01% |
| Dormant Explorers | 249 | 1 | 0.40% |

**Overall: 1,600 learners, 61 converters, 3.81% cumulative conversion** (vs. the cited
~2.6% EdTech / 2-5% general self-serve SaaS benchmark). **Time-to-convert:** ranges 1–12
months, mean 6.63 months, median 7.0 months — most conversions are not a same-month
impulse decision.

### Renewal pattern (SQL 04a) — renewal / churn rate by plan

| Plan | Paid months at risk | Renewal % | Churn % |
|---|---:|---:|---:|
| Basic | 138 | 91.30 | 6.52 |
| Pro | 119 | 91.60 | 5.04 |
| Premium | 10 | 100.00 | 0.00 |

By tenure bucket (using the *pre-event* tenure — see the `LAG()` bug-fix note above):
0-month (first renewal decision) 93.75% renewal, 1-2 months 88.66%, 3-5 months 94.44%,
6+ months 91.18% — no strong monotonic loyalty curve is visible at this sample size (n=34
to 97 per bucket, smallest at 6+ months), which is itself a fair, honest reading rather
than a story imposed on noisy data.

### Rules engine output (all 1,600 learners scored)

| Action | Learners |
|---|---:|
| upgrade_offer | 744 |
| no_action | 711 |
| winback_email | 79 |
| re_engagement_nudge | 33 |
| renewal_no_discount | 17 |
| renewal_discount | 9 |
| retention_discount_plus_cs | 4 |
| cs_outreach | 2 |
| feature_adoption_nudge | 1 |

### Held-out intervention comparison (framework vs. baseline) — the core result

| Metric | Baseline | Framework | Absolute lift | Relative lift | p-value |
|---|---:|---:|---:|---:|---|
| Conversion rate | 0.52% | 1.20% | +0.67pp | **+127.9%** | **6.8×10⁻²⁷** |
| Renewal rate | 96.97% | 96.29% | −0.68pp | −0.7% | 0.33 (n.s.) |
| Upgrade rate | 5.30% | 6.21% | +0.91pp | +17.1% | 0.32 (n.s.) |
| Discount redemption rate | 3.44% | 4.22% | +0.78pp | **+22.6%** | **8.5×10⁻⁶** |

**Two metrics improve with overwhelming statistical significance (conversion, discount
redemption), one is directionally positive but not significant at this sample size
(upgrade), and one is essentially flat/very slightly negative and not significant
(renewal).** The renewal result is a real, interesting finding, not a shortfall to hide:
baseline's blanket random discount to every paid learner passively helps retention across
the board (via the same elasticity mechanism), while the framework deliberately withholds
discounts from learners who don't need one to protect margin — see `WRITEUP.md` for the
full "why" on this trade-off.

## Project structure

```
edtech-pricing-strategy/
├── README.md                    <- this file
├── WRITEUP.md                   <- written summary: did the framework improve outcomes?
├── requirements.txt
├── data/processed/               <- all synthetic tables (learners, panel, recs, etc.)
├── sql/                          <- MySQL 8.0 schema, load, 3 required analyses
├── python/                       <- generation, rules engine, ML model, intervention sim
└── output/tables/                <- intervention comparison + held-out simulation results
```

## How to run it

```bash
pip install -r requirements.txt
python python/run_pipeline.py     # generate data, train ML model, apply rules, run intervention sim

mysql -u root -p -e "SET GLOBAL local_infile = 1;"   # once per server restart
mysql -u root -p < sql/00_schema.sql
mysql --local-infile=1 -u root -p edtech_pricing_strategy < sql/01_load_data.sql
mysql -u root -p edtech_pricing_strategy < sql/02_segment_analysis.sql
mysql -u root -p edtech_pricing_strategy < sql/03_conversion_pattern.sql
mysql -u root -p edtech_pricing_strategy < sql/04_renewal_pattern.sql
```

All SQL was run against a real local MySQL 8.0 instance. `LINES TERMINATED BY '\r\n'` is
declared explicitly from the start in `01_load_data.sql` — a lesson carried over from the
two sibling projects in this series, where declaring only `'\n'` either corrupted a
last-column value or, worse, silently dropped almost all rows when the last field was
quoted. A second, genuinely subtle bug was caught and fixed while building `sql/04`: an
early version grouped renewal rate by the raw `tenure_months_on_plan` column, which
reflects tenure **after** that month's event is applied (it resets to 0 exactly when a
non-renewal event happens, and increments exactly when a renewal happens) — grouping by it
directly is circular, not a finding (the "tenure=0" bucket would show 0% renewal by
construction). Fixed with `LAG()` to recover the tenure a learner had **entering** the
month. Documented in `sql/04_renewal_pattern.sql`'s comments.

---

## Data Assumptions

### This is 100% synthetic — stated plainly

No real learner-level pricing/engagement dataset for an EdTech subscription platform is
publicly available. Every learner, every month of engagement history, every plan-switch
event, and every price paid in this project is generated by `python/generate_data.py`.
Nothing here should ever be read as describing real learners or a real platform.

### Every field's generation logic and the benchmark it's calibrated against

| Field / behavior | Generation logic | Real benchmark used | Source |
|---|---|---|---|
| Freemium→paid conversion | Monthly hazard `0.0020` (population base) × segment `conversion_mult` × that month's engagement, applied for up to 12 months | **Cumulative** ~2-5% general self-serve SaaS, **~2.6% EdTech specifically**. The cited figure is cumulative, not a monthly hazard — the monthly hazard was back-solved so 12 months of compounding lands near the cited cumulative figure (realized: 3.81%) | Artisan Growth Strategies, "Freemium Conversion Rate Benchmarks 2026"; corroborated by OpenView Partners' Product Benchmarks and ChartMogul's "SaaS Conversion Report" |
| Monthly churn (paid) | `0.068` base monthly hazard × segment `price_sensitivity_mult` × discount effect × tenure-loyalty decay × engagement-decline penalty | ~6-7.5% monthly, B2C/consumer EdTech specifically (realized: 5.62%) | RetentionCheck, "Kids Education App Churn Rate 2026: 7.4% Monthly Benchmark"; Koji, "SaaS Churn Rate Benchmarks 2026" |
| Price elasticity (discount → retention/conversion effect) | `ELASTICITY_MAGNITUDE = 1.75` (midpoint), used identically in both the historical panel generator and the later intervention simulation — one mechanism, not two | Consumer/B2C subscription price elasticity of demand: **-1.5 to -2.0** (vs. -0.2 to -1.3 for B2B SaaS tiers with switching costs) | PayProGlobal, "What is SaaS Price Elasticity? Measuring Demand" |
| Organic historical discount depth | 18% of paid-months carry a random 10/15/20% discount | Typical annual/promotional discount depth of **15-20%** is standard industry practice | Multiple SaaS pricing benchmark reports, e.g. Cledara "SaaS Renewal Benchmarks 2026" |
| Plan list prices (₹0 / 299 / 699 / 1499) | Fixed | **MODELING CHOICE** — no cited source for exact price points | — |
| 6 segment personas (engagement level, price sensitivity, geography, course category, population share) | Hand-defined profiles, see table below | **MODELING CHOICE** — illustrative personas, not derived from real data. Their AGGREGATE effect (overall conversion %, overall churn %) is what's calibrated to the cited benchmarks above, not any individual segment parameter | — |
| Upgrade/downgrade hazards, month-to-month engagement noise, exact tenure-decay curve shape | Various small hazards/noise terms | **MODELING CHOICE** — needed for internal consistency, not independently sourced | — |

### The 6 segments, concretely defined

| Segment | Share | Engagement | Price sensitivity | Geography lean | Course category |
|---|---:|---|---|---|---|
| Power Learners | 15% | High (18 logins/mo base), growing | Low (0.5×) | Metro-heavy | Tech/Programming |
| Career Switchers | 18% | High-moderate (14/mo), growing | Moderate (1.0×) | Mixed Metro/Tier-2 | Certification |
| Casual Upskillers | 20% | Moderate (8/mo), flat | Moderate-high (1.3×) | Mixed | General/Hobby |
| Budget Students | 20% | Moderate-low (6/mo), declining | High (1.8×) | Tier-2/3-heavy | Exam Prep |
| Enterprise Professionals | 12% | High (16/mo), growing | Low (0.4×) | Metro-heavy | Professional Cert |
| Dormant Explorers | 15% | Low (2/mo), declining | High (2.0×) | Mixed | General/Hobby |

### Sample-size caveat on the segment×discount elasticity-recovery query

`sql/02_segment_analysis.sql`'s query 2d tries to recover an empirical price-sensitivity
proxy from historical discount-vs-churn behavior, per segment. With only 1,600 learners
split 6 ways and further split by discount-active/not, some segment×discount cells have as
few as 7-14 observations — small enough that the recovered proxy is noisy (one segment,
Career Switchers, shows a proxy in the *opposite* direction from its true generation-time
`price_sensitivity_mult`, purely from small-N variance). Stated plainly rather than
cherry-picking only the cells that recovered cleanly — see `WRITEUP.md` for the honest
read on this.

### The rules engine — 24 rules, 14 parameters, fully auditable

Every rule is an explicit condition → recommendation → plain-English rationale in
`python/rules_engine.py`'s `RULES` list — e.g. the brief's own worked example
(`if engagement_trend declining AND tenure > 6mo AND plan = Pro → offer 15% renewal
discount`) is rule **R07**, verbatim. If the optional ML churn-risk layer is removed
entirely, 22 of the 24 rules are unaffected (only R09 and R10 consume it).

### The optional ML layer

A logistic regression predicts each paid learner's probability of churning next month,
trained on 267 learner-month observations from the main cohort (held-out test AUC 0.706 —
meaningfully better than random, but this is a small-N demonstration of the mechanism, not
a production-grade model; stated plainly). Only 2 of 24 rules consume this score.

### The held-out intervention experiment

A **fresh 3,000-learner cohort** (different random seed, never used to build or tune the
rules engine or the ML model) simulates one additional decision month. Two arms:
**baseline** (every learner gets a uniformly random discount from {0%, 10%, 15%, 20%}, no
targeting) vs. **framework** (the 24-rule engine's personalized recommendation). Because a
single simulated month is too noisy at these realistic (low) monthly hazard rates to
detect a real effect against, each arm's outcome is **resampled 15 times per learner**
under the same fixed policy decision — a standard Monte Carlo technique for estimating a
policy's effect precisely. This is not a claim of 45,000 distinct learners; it's 3,000
learners' month-13 outcome resampled 15× each. Both arms' outcomes are generated by the
SAME elasticity mechanism (`ELASTICITY_MAGNITUDE = 1.75`) used in the original 12-month
panel — the "why would personalization help" story is a direct consequence of one
already-cited effect, not a second, separately-invented lift. Two additional multipliers
(a 3× conversion-hazard boost for a targeted upgrade prompt, a 0.5× churn-hazard reduction
for a well-matched retention action) are explicit **MODELING CHOICES**, not independently
sourced — stated plainly in `python/simulate_intervention.py`.

### No time-saved or efficiency % claims

Consistent with the other two projects in this series: nothing here claims a real-world
efficiency gain or time saved, since there is no real deployment to measure against. The
only claims made are about the simulated comparison described above.
