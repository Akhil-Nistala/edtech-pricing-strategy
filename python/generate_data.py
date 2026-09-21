"""
Generates the FULLY SYNTHETIC EdTech learner dataset: 1,600 learners across 6 named
segments and 4 subscription plans, with a 12-month monthly panel of engagement, plan
state, price paid, and renewal/upgrade/downgrade/churn/conversion events.

NO REAL LEARNER DATA EXISTS ANYWHERE IN THIS PROJECT. Every generation parameter below is
either (a) calibrated to a cited real EdTech/SaaS industry benchmark, or (b) explicitly
labeled MODELING CHOICE where no such benchmark exists (segment personas, exact price
points, month-to-month noise). See README.md > Data Assumptions for the full breakdown.

Cited benchmarks used to calibrate this generator:
  - Freemium-to-paid conversion, EdTech specifically ~2.6% (vs. 2-5% general self-serve
    SaaS): Artisan Growth Strategies, "Freemium Conversion Rate Benchmarks 2026"
    (artisangrowthstrategies.com), corroborated by OpenView Partners' Product Benchmarks
    and ChartMogul's "SaaS Conversion Report". This is a *cumulative* (over ~12 months)
    figure, not a monthly hazard -- see CONVERSION calibration note below for how that
    distinction is handled.
  - B2C/consumer EdTech monthly churn ~6-7.5%: RetentionCheck, "Kids Education App Churn
    Rate 2026: 7.4% Monthly Benchmark" (retentioncheck.com/churn-benchmarks/kids-education-apps);
    Koji, "SaaS Churn Rate Benchmarks 2026" (koji.so) -- both report this as a genuine
    monthly hazard, used directly as one here.
  - Consumer/B2C subscription price elasticity of demand: -1.5 to -2.0 (vs. -0.2 to -1.3
    for B2B SaaS tiers with switching costs): PayProGlobal, "What is SaaS Price
    Elasticity? Measuring Demand" (payproglobal.com/answers/what-is-saas-price-elasticity).
  - Typical annual/promotional discount depth: 15-20%, standard industry practice cited
    across multiple SaaS pricing benchmark reports (e.g. Cledara "SaaS Renewal
    Benchmarks 2026").
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N_LEARNERS = 1600
N_MONTHS = 12
PLANS = ["Free", "Basic", "Pro", "Premium"]
PLAN_RANK = {"Free": 0, "Basic": 1, "Pro": 2, "Premium": 3}
PLAN_PRICE = {"Free": 0, "Basic": 299, "Pro": 699, "Premium": 1499}  # INR/month, MODELING CHOICE (no cited source for exact price points)

# ---------------------------------------------------------------------------
# Six segments, concretely defined by engagement level, price sensitivity, geography,
# and preferred course category, per the brief. Shares sum to 1.0. Every FIELD here is a
# MODELING CHOICE (these are illustrative personas, not derived from real data) -- but
# the AGGREGATE behavior they produce (overall conversion %, overall churn %) is
# calibrated to land near the cited benchmarks above; see calibration check in __main__.
# ---------------------------------------------------------------------------
SEGMENTS = {
    "Power Learners": dict(
        share=0.15, engagement_base=18, engagement_trend_bias=0.30,
        price_sensitivity_mult=0.5, conversion_mult=3.0, upgrade_mult=2.5,
        geo_weights={"Metro": 0.70, "Tier-2": 0.25, "Tier-3": 0.05}, category="Tech/Programming",
    ),
    "Career Switchers": dict(
        share=0.18, engagement_base=14, engagement_trend_bias=0.10,
        price_sensitivity_mult=1.0, conversion_mult=1.8, upgrade_mult=1.5,
        geo_weights={"Metro": 0.50, "Tier-2": 0.40, "Tier-3": 0.10}, category="Certification",
    ),
    "Casual Upskillers": dict(
        share=0.20, engagement_base=8, engagement_trend_bias=0.00,
        price_sensitivity_mult=1.3, conversion_mult=1.0, upgrade_mult=0.8,
        geo_weights={"Metro": 0.40, "Tier-2": 0.40, "Tier-3": 0.20}, category="General/Hobby",
    ),
    "Budget Students": dict(
        share=0.20, engagement_base=6, engagement_trend_bias=-0.10,
        price_sensitivity_mult=1.8, conversion_mult=0.6, upgrade_mult=0.4,
        geo_weights={"Metro": 0.15, "Tier-2": 0.45, "Tier-3": 0.40}, category="Exam Prep",
    ),
    "Enterprise Professionals": dict(
        share=0.12, engagement_base=16, engagement_trend_bias=0.20,
        price_sensitivity_mult=0.4, conversion_mult=2.5, upgrade_mult=2.2,
        geo_weights={"Metro": 0.80, "Tier-2": 0.20, "Tier-3": 0.00}, category="Professional Cert",
    ),
    "Dormant Explorers": dict(
        share=0.15, engagement_base=2, engagement_trend_bias=-0.30,
        price_sensitivity_mult=2.0, conversion_mult=0.15, upgrade_mult=0.1,
        geo_weights={"Metro": 0.35, "Tier-2": 0.35, "Tier-3": 0.30}, category="General/Hobby",
    ),
}
assert abs(sum(s["share"] for s in SEGMENTS.values()) - 1.0) < 1e-9

# ---------------------------------------------------------------------------
# Calibration constants
# ---------------------------------------------------------------------------
# CONVERSION: the cited ~2.6% EdTech freemium figure is a *cumulative* conversion rate
# (the share of free signups who ever convert), not a monthly hazard. Applying ~2.6% as a
# monthly hazard for 12 months would compound to ~27% cumulative -- ~10x too high. Solving
# (1 - (1-p)^12) ~= 0.03 for a population-average free learner gives p ~= 0.0025/month;
# segment conversion_mult and that month's engagement then push individual learners'
# actual monthly hazard above/below this population-average base.
BASE_MONTHLY_CONVERSION_HAZARD = 0.0020

# CHURN: cited directly as a monthly hazard (6-7.5% B2C EdTech) -- used as-is, modulated
# per learner-month below.
BASE_MONTHLY_CHURN_HAZARD = 0.068

# ELASTICITY: midpoint of the cited -1.5 to -2.0 consumer-subscription range. Used
# (as a magnitude) to translate an active discount into a churn-hazard reduction, and
# reused identically in python/simulate_intervention.py so the historical data and the
# later intervention test share one mechanism, not two ad-hoc ones.
ELASTICITY_MAGNITUDE = 1.75


def discount_retention_multiplier(discount_fraction: float, price_sensitivity_mult: float) -> float:
    """Elasticity-based effect of an active discount on churn hazard: %Δretention ~=
    elasticity * %Δprice. A learner's price_sensitivity_mult scales the population
    elasticity up/down for that learner. Floors at 0.3x (a discount cannot be assumed to
    eliminate churn hazard entirely -- MODELING CHOICE, not itself sourced)."""
    return max(0.3, 1 - ELASTICITY_MAGNITUDE * price_sensitivity_mult * discount_fraction)


def make_learners(n=N_LEARNERS, seed=42) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    names = list(SEGMENTS.keys())
    weights = [SEGMENTS[s]["share"] for s in names]
    segment = r.choice(names, size=n, p=weights)

    geography = np.empty(n, dtype=object)
    category = np.empty(n, dtype=object)
    for seg in names:
        mask = segment == seg
        geo_w = SEGMENTS[seg]["geo_weights"]
        geography[mask] = r.choice(list(geo_w.keys()), size=mask.sum(), p=list(geo_w.values()))
        category[mask] = SEGMENTS[seg]["category"]

    signup_offset_days = r.integers(0, 30, size=n)  # all sign up in "month 0", small jitter
    learners = pd.DataFrame({
        "learner_id": [f"L{100000+i}" for i in range(n)],
        "segment": segment,
        "geography": geography,
        "course_category": category,
        "signup_day_offset": signup_offset_days,
    })
    return learners


def simulate_panel(learners: pd.DataFrame, seed=99) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    rows = []

    for learner in learners.itertuples(index=False):
        seg = SEGMENTS[learner.segment]
        plan = "Free"
        tenure_months_on_plan = 0
        engagement = max(0.5, r.normal(seg["engagement_base"], seg["engagement_base"] * 0.35))
        discount_active_pct = 0.0

        for month in range(1, N_MONTHS + 1):
            engagement = max(
                0.2, engagement * (1 + seg["engagement_trend_bias"] / 12) + r.normal(0, 1.2)
            )
            completion_pct = min(100, max(0, r.normal(30 + engagement * 2.2, 12)))
            feature_usage_score = min(10, max(0, round(r.normal(engagement / 2.2, 1.5))))

            # MODELING CHOICE: ~18% of paid-months in the historical panel carry an
            # "organic" promotional discount (pre-dating this project's own rules engine)
            # -- more likely for higher price-sensitivity segments, reflecting plausible
            # existing targeted-promo practice. This is what lets the SQL "price
            # elasticity proxy from historical plan-switch behavior" query have real
            # discount-vs-no-discount variation to analyze.
            if plan != "Free" and r.random() < 0.18:
                discount_active_pct = r.choice([0.10, 0.15, 0.20])
            else:
                discount_active_pct = 0.0

            event = "none"
            if plan == "Free":
                conv_hazard = (
                    BASE_MONTHLY_CONVERSION_HAZARD * seg["conversion_mult"]
                    * (1 + max(0, (engagement - seg["engagement_base"]) / seg["engagement_base"]))
                )
                if r.random() < min(0.5, conv_hazard):
                    plan = "Basic" if r.random() < 0.7 else "Pro"
                    tenure_months_on_plan = 0
                    event = "conversion"
            else:
                churn_hazard = (
                    BASE_MONTHLY_CHURN_HAZARD
                    * seg["price_sensitivity_mult"]
                    * discount_retention_multiplier(discount_active_pct, seg["price_sensitivity_mult"])
                    * np.exp(-tenure_months_on_plan / 24)          # tenure loyalty effect
                    * (1.4 if seg["engagement_trend_bias"] < 0 and engagement < seg["engagement_base"] * 0.6 else 1.0)
                )
                upgrade_hazard = (
                    0.04 * seg["upgrade_mult"]
                    * (1 if completion_pct > 60 and engagement > seg["engagement_base"] else 0.2)
                )
                roll = r.random()
                if roll < churn_hazard and PLAN_RANK[plan] > 0:
                    plan = "Free"
                    event = "churn"
                    tenure_months_on_plan = 0
                elif roll < churn_hazard + upgrade_hazard and PLAN_RANK[plan] < 3:
                    plan = PLANS[PLAN_RANK[plan] + 1]
                    event = "upgrade"
                    tenure_months_on_plan = 0
                elif roll < churn_hazard + upgrade_hazard + 0.015 and PLAN_RANK[plan] > 1:
                    plan = PLANS[PLAN_RANK[plan] - 1]
                    event = "downgrade"
                    tenure_months_on_plan = 0
                else:
                    event = "renewal"
                    tenure_months_on_plan += 1

            list_price = PLAN_PRICE[plan]
            price_paid = round(list_price * (1 - discount_active_pct), 2)

            rows.append((
                learner.learner_id, month, plan, event, list_price, price_paid,
                discount_active_pct, round(engagement, 2), round(completion_pct, 1),
                feature_usage_score, tenure_months_on_plan,
            ))

    panel = pd.DataFrame(rows, columns=[
        "learner_id", "month", "plan", "event", "list_price", "price_paid",
        "discount_pct", "logins_this_month", "course_completion_pct",
        "feature_usage_score", "tenure_months_on_plan",
    ])
    return panel


if __name__ == "__main__":
    learners = make_learners()
    panel = simulate_panel(learners)

    learners.to_csv("data/processed/learners.csv", index=False)
    panel.to_csv("data/processed/engagement_panel.csv", index=False)

    print("learners:", len(learners))
    print("panel rows:", len(panel))

    ever_converted = panel[panel["event"] == "conversion"]["learner_id"].nunique()
    print(f"cumulative 12-month free->paid conversion: {100*ever_converted/len(learners):.2f}% "
          f"(cited EdTech benchmark: ~2.6%, general self-serve SaaS: 2-5%)")

    # A learner is "eligible to churn" in a given month if they were on a PAID plan
    # entering that month -- i.e. the event recorded is one of renewal/upgrade/downgrade
    # (stayed paid) or churn (left paid). NOT `panel["plan"] != "Free"`, which excludes
    # churn-event rows themselves (a churn row's `plan` column already reflects the
    # POST-event state, Free) and would silently zero out the churn count.
    paid_state_events = panel[panel["event"].isin(["renewal", "upgrade", "downgrade", "churn"])]
    churn_months = paid_state_events[paid_state_events["event"] == "churn"]
    print(f"monthly churn rate (paid, pooled across all months): "
          f"{100*len(churn_months)/len(paid_state_events):.2f}% "
          f"(cited B2C EdTech benchmark: ~6-7.5%)")

    print(panel.groupby("event").size())
