"""
The pricing/retention recommendation rules engine: 24 explicit, auditable business rules
evaluated over 14 behavioral/pricing parameters per learner. Deterministic and fully
transparent by design (per the brief: "don't leave the AI framing vague") -- an optional
ML churn-risk score (python/ml_churn_model.py) is ONE of the 14 input parameters two rules
consume, not a black box driving the whole system. If the ML layer were removed, 22 of the
24 rules still function unchanged.

Each rule is a row in RULES below: an explicit condition function over a context dict,
plus the recommendation it produces and a plain-English rationale. Rules are evaluated in
priority order; the FIRST matching rule wins (standard deterministic rules-engine
pattern). Rule R24 is an always-true fallback so every learner gets a recommendation.

The 14 parameters a rule can condition on (built by `build_context`):
  1. current_plan            8.  logins_this_month
  2. segment                 9.  discount_pct_active
  3. price_sensitivity_mult  10. months_since_signup
  4. engagement_trend        11. churn_risk_score (ML, optional)
  5. tenure_months_on_plan   12. geography
  6. course_completion_pct   13. course_category
  7. feature_usage_score     14. consecutive_zero_login_months
"""
import numpy as np
import pandas as pd

from generate_data import PLAN_RANK, PLANS, SEGMENTS


def build_context(learner_row, learner_panel: pd.DataFrame, churn_risk_score: float | None) -> dict:
    """learner_panel: this learner's 12 monthly rows, in month order."""
    last = learner_panel.iloc[-1]
    recent3 = learner_panel.tail(3)["logins_this_month"].mean()
    prior3 = learner_panel.iloc[-6:-3]["logins_this_month"].mean() if len(learner_panel) >= 6 else recent3

    if recent3 > prior3 * 1.1:
        trend = "growing"
    elif recent3 < prior3 * 0.9:
        trend = "declining"
    else:
        trend = "stable"

    zero_login_streak = 0
    for logins in learner_panel["logins_this_month"].values[::-1]:
        if logins < 1:
            zero_login_streak += 1
        else:
            break

    return {
        "learner_id": learner_row["learner_id"],
        "current_plan": last["plan"],
        "segment": learner_row["segment"],
        "price_sensitivity_mult": SEGMENTS[learner_row["segment"]]["price_sensitivity_mult"],
        "engagement_trend": trend,
        "tenure_months_on_plan": int(last["tenure_months_on_plan"]),
        "course_completion_pct": float(last["course_completion_pct"]),
        "feature_usage_score": float(last["feature_usage_score"]),
        "logins_this_month": float(last["logins_this_month"]),
        "discount_pct_active": float(last["discount_pct"]),
        "months_since_signup": int(last["month"]),
        "churn_risk_score": churn_risk_score,
        "geography": learner_row["geography"],
        "course_category": learner_row["course_category"],
        "consecutive_zero_login_months": zero_login_streak,
    }


def _next_plan(current: str) -> str:
    return PLANS[min(3, PLAN_RANK[current] + 1)]


def _prev_plan(current: str) -> str:
    return PLANS[max(0, PLAN_RANK[current] - 1)]


RULES = [
    dict(id="R01", desc="Free, high completion+engagement, any tenure -> upsell w/ small discount",
         cond=lambda c: c["current_plan"] == "Free" and c["course_completion_pct"] >= 70 and c["logins_this_month"] >= 15,
         action="upgrade_offer", target_plan="Pro", discount=0.10,
         rationale="High-intent free user showing strong completion and engagement -- ready for a paid upsell."),
    dict(id="R02", desc="Free, moderate engagement, low-price-sensitivity segment -> upsell, no discount",
         cond=lambda c: c["current_plan"] == "Free" and c["course_completion_pct"] >= 50 and c["logins_this_month"] >= 10 and c["segment"] in ("Power Learners", "Enterprise Professionals"),
         action="upgrade_offer", target_plan="Basic", discount=0.0,
         rationale="Segment historically converts well without discounting; a discount here would erode margin unnecessarily."),
    dict(id="R03", desc="Free, moderate engagement, price-sensitive -> upsell w/ discount",
         cond=lambda c: c["current_plan"] == "Free" and c["course_completion_pct"] >= 50 and c["logins_this_month"] >= 10 and c["price_sensitivity_mult"] >= 1.3,
         action="upgrade_offer", target_plan="Basic", discount=0.15,
         rationale="Price-sensitive segment; a modest discount meaningfully lowers the conversion barrier per the elasticity model."),
    dict(id="R04", desc="Free, very low engagement, some tenure -> re-engagement nudge, no discount",
         cond=lambda c: c["current_plan"] == "Free" and c["course_completion_pct"] < 20 and c["logins_this_month"] < 3 and c["months_since_signup"] >= 3,
         action="re_engagement_nudge", target_plan=None, discount=0.0,
         rationale="Low engagement free user unlikely to respond to a price incentive; a feature/content nudge addresses the real barrier."),
    dict(id="R05", desc="Free, dormant 2+ months -> win-back, no discount",
         cond=lambda c: c["current_plan"] == "Free" and c["consecutive_zero_login_months"] >= 2,
         action="winback_email", target_plan=None, discount=0.0,
         rationale="Fully dormant; a discount is wasted spend until engagement resumes."),
    dict(id="R06", desc="Basic, declining engagement, tenure>=6mo -> renewal discount 15% (brief's own example)",
         cond=lambda c: c["current_plan"] == "Basic" and c["engagement_trend"] == "declining" and c["tenure_months_on_plan"] >= 6,
         action="renewal_discount", target_plan="Basic", discount=0.15,
         rationale="Declining engagement + established tenure signals renewal risk worth a moderate discount (this is the brief's own worked example)."),
    dict(id="R07", desc="Pro, declining engagement, tenure>=6mo -> renewal discount 15%",
         cond=lambda c: c["current_plan"] == "Pro" and c["engagement_trend"] == "declining" and c["tenure_months_on_plan"] >= 6,
         action="renewal_discount", target_plan="Pro", discount=0.15,
         rationale="Same declining-engagement + tenure pattern as R06, scaled to the Pro tier."),
    dict(id="R08", desc="Premium, declining engagement, price-sensitive -> retention discount 20%",
         cond=lambda c: c["current_plan"] == "Premium" and c["engagement_trend"] == "declining" and c["price_sensitivity_mult"] >= 1.3,
         action="retention_discount", target_plan="Premium", discount=0.20,
         rationale="Highest-value at-risk learner with meaningful price sensitivity -- the largest discount tier is justified by the LTV at stake."),
    dict(id="R09", desc="Any paid, high ML churn risk, low price sensitivity -> CS outreach, no discount",
         cond=lambda c: c["current_plan"] != "Free" and (c["churn_risk_score"] or 0) >= 0.6 and c["price_sensitivity_mult"] < 1.0,
         action="cs_outreach", target_plan=None, discount=0.0,
         rationale="Low price sensitivity means discounting is unlikely to be the effective retention lever; a human retention conversation is."),
    dict(id="R10", desc="Any paid, high ML churn risk, price-sensitive -> retention discount 20% + CS flag",
         cond=lambda c: c["current_plan"] != "Free" and (c["churn_risk_score"] or 0) >= 0.6 and c["price_sensitivity_mult"] >= 1.0,
         action="retention_discount_plus_cs", target_plan=None, discount=0.20,
         rationale="Combines the ML risk signal with a price-sensitivity-matched offer -- both the behavioral and pricing lever point the same way."),
    dict(id="R11", desc="Basic, strong+growing usage -> upgrade to Pro, no discount",
         cond=lambda c: c["current_plan"] == "Basic" and c["course_completion_pct"] >= 70 and c["engagement_trend"] == "growing" and c["tenure_months_on_plan"] >= 2,
         action="upgrade_offer", target_plan="Pro", discount=0.0,
         rationale="Strong, growing usage signals upgrade readiness without needing a discount."),
    dict(id="R12", desc="Pro, strong+growing usage -> upgrade to Premium, small discount",
         cond=lambda c: c["current_plan"] == "Pro" and c["course_completion_pct"] >= 70 and c["engagement_trend"] == "growing" and c["tenure_months_on_plan"] >= 2,
         action="upgrade_offer", target_plan="Premium", discount=0.05,
         rationale="Top-tier upgrade benefits from a small nudge to close, even for an engaged learner."),
    dict(id="R13", desc="Basic, price-sensitive, decent usage -> upgrade to Pro w/ discount",
         cond=lambda c: c["current_plan"] == "Basic" and c["price_sensitivity_mult"] >= 1.3 and c["course_completion_pct"] >= 60 and c["tenure_months_on_plan"] >= 2,
         action="upgrade_offer", target_plan="Pro", discount=0.15,
         rationale="Price-sensitive learner with real usage; the discount offsets the price barrier to upgrading."),
    dict(id="R14", desc="Any paid, sharp completion decline + short tenure -> preemptive downgrade offer",
         cond=lambda c: c["current_plan"] != "Free" and c["engagement_trend"] == "declining" and c["course_completion_pct"] < 30 and c["tenure_months_on_plan"] < 3,
         action="downgrade_offer", target_plan="prev", discount=0.0,
         rationale="Offering a smaller plan preserves some revenue and the relationship versus losing the learner to full churn."),
    dict(id="R15", desc="Premium, low usage, tenure>=3mo -> downgrade offer to Pro",
         cond=lambda c: c["current_plan"] == "Premium" and c["course_completion_pct"] < 30 and c["logins_this_month"] < 5 and c["tenure_months_on_plan"] >= 3,
         action="downgrade_offer", target_plan="Pro", discount=0.0,
         rationale="Premium features aren't being used; right-sizing to Pro protects the relationship better than an eventual full churn."),
    dict(id="R16", desc="Any paid, renewal point, low price sensitivity, healthy engagement -> renew, no discount",
         cond=lambda c: c["current_plan"] != "Free" and c["price_sensitivity_mult"] < 0.7 and c["engagement_trend"] in ("stable", "growing"),
         action="renewal_no_discount", target_plan=None, discount=0.0,
         rationale="Low price sensitivity + healthy engagement: discounting here only erodes margin with no retention benefit."),
    dict(id="R17", desc="Any paid, renewal point, moderate price sensitivity, stable engagement -> small renewal discount",
         cond=lambda c: c["current_plan"] != "Free" and 0.7 <= c["price_sensitivity_mult"] < 1.3 and c["engagement_trend"] == "stable",
         action="renewal_discount", target_plan=None, discount=0.10,
         rationale="Moderate price sensitivity; a small discount pre-empts price-driven churn per the -1.75 elasticity model this project uses."),
    dict(id="R18", desc="Free, Budget Students segment, decent completion -> entry-tier upsell only, deep discount",
         cond=lambda c: c["current_plan"] == "Free" and c["segment"] == "Budget Students" and c["course_completion_pct"] >= 40,
         action="upgrade_offer", target_plan="Basic", discount=0.15,
         rationale="Segment is historically price-sensitive and budget-constrained; recommend the accessible entry tier, not a premium upsell likely to fail."),
    dict(id="R19", desc="Free, Dormant Explorers segment -> no targeted action",
         cond=lambda c: c["current_plan"] == "Free" and c["segment"] == "Dormant Explorers",
         action="no_action", target_plan=None, discount=0.0,
         rationale="Segment has near-zero historical conversion propensity; discount budget has the lowest expected return here."),
    dict(id="R20", desc="Any paid, already discounted, stable/growing -> no additional discount at renewal",
         cond=lambda c: c["current_plan"] != "Free" and c["discount_pct_active"] > 0 and c["engagement_trend"] in ("stable", "growing"),
         action="renewal_no_discount", target_plan=None, discount=0.0,
         rationale="Already receiving a promotional rate; stacking a further discount is unnecessary margin erosion."),
    dict(id="R21", desc="Any paid, low feature adoption despite tenure -> education nudge, no discount",
         cond=lambda c: c["current_plan"] != "Free" and c["feature_usage_score"] <= 2 and c["tenure_months_on_plan"] >= 3,
         action="feature_adoption_nudge", target_plan=None, discount=0.0,
         rationale="Low feature adoption despite tenure suggests an onboarding/education gap, not a price problem."),
    dict(id="R22", desc="Free, Tier-3 geography, decent completion -> upsell w/ deepest discount tier",
         cond=lambda c: c["current_plan"] == "Free" and c["geography"] == "Tier-3" and c["course_completion_pct"] >= 40,
         action="upgrade_offer", target_plan="Basic", discount=0.20,
         rationale="Tier-3 geography correlates with lower ability-to-pay in this market; a deeper discount matches willingness-to-pay while still converting."),
    dict(id="R23", desc="Exam Prep category, early tenure, decent completion -> time-boxed upgrade offer",
         cond=lambda c: c["course_category"] == "Exam Prep" and c["months_since_signup"] <= 2 and c["course_completion_pct"] >= 50,
         action="upgrade_offer", target_plan="upgrade_or_basic", discount=0.10,
         rationale="Exam-prep learners convert on urgency; a time-boxed offer matches their planning horizon."),
    dict(id="R24", desc="Fallback: no other rule matched -> monitor, no action",
         cond=lambda c: True,
         action="no_action", target_plan=None, discount=0.0,
         rationale="Insufficient signal to justify a targeted offer this cycle; default to observation."),
]


def apply_rules(ctx: dict) -> dict:
    for rule in RULES:
        if rule["cond"](ctx):
            target = rule["target_plan"]
            if target == "prev":
                target = _prev_plan(ctx["current_plan"])
            elif target == "upgrade_or_basic":
                target = _next_plan(ctx["current_plan"]) if ctx["current_plan"] != "Free" else "Basic"
            return {
                "learner_id": ctx["learner_id"],
                "rule_id": rule["id"],
                "action": rule["action"],
                "target_plan": target,
                "discount_pct": rule["discount"],
                "rationale": rule["rationale"],
            }
    raise RuntimeError("no rule matched -- R24 fallback should always match")


def recommend_for_all(learners: pd.DataFrame, panel: pd.DataFrame, churn_scores: dict[str, float] | None = None) -> pd.DataFrame:
    churn_scores = churn_scores or {}
    recs = []
    panel_by_learner = {lid: g.sort_values("month") for lid, g in panel.groupby("learner_id")}
    for row in learners.itertuples(index=False):
        ctx = build_context(row._asdict(), panel_by_learner[row.learner_id], churn_scores.get(row.learner_id))
        recs.append(apply_rules(ctx))
    return pd.DataFrame(recs)


if __name__ == "__main__":
    learners = pd.read_csv("data/processed/learners.csv")
    panel = pd.read_csv("data/processed/engagement_panel.csv")
    recs = recommend_for_all(learners, panel)
    recs.to_csv("data/processed/recommendations.csv", index=False)
    print(recs["rule_id"].value_counts())
    print(recs["action"].value_counts())
