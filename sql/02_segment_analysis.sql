-- ============================================================================
-- Segment-level analysis -- SYNTHETIC data throughout (see README > Data Assumptions).
-- renewal / upgrade / downgrade / churn rate per segment (and per segment x plan), plus
-- a price-sensitivity / elasticity PROXY recovered from historical discount-vs-outcome
-- behavior (not the true generation-time elasticity itself -- this is the empirical
-- signal an analyst would actually have to work with).
-- ============================================================================

USE edtech_pricing_strategy;

-- 2a. plan_at_start_of_month: the plan a learner was on BEFORE that month's event.
-- engagement_panel.plan reflects the plan AFTER the event, so an upgrade/downgrade/churn
-- row's `plan` column is already the new state -- LAG() recovers the prior state; month 1
-- always starts from Free (the generator's actual initial condition).
WITH plan_before AS (
    SELECT
        learner_id, month, event, discount_pct,
        COALESCE(LAG(plan) OVER (PARTITION BY learner_id ORDER BY month), 'Free') AS plan_at_start
    FROM engagement_panel
)

-- 2b. Renewal / upgrade / downgrade / churn rate by segment
SELECT
    l.segment,
    COUNT(*) AS paid_months_at_risk,
    ROUND(100.0 * SUM(pb.event = 'renewal') / COUNT(*), 2) AS renewal_rate_pct,
    ROUND(100.0 * SUM(pb.event = 'upgrade') / COUNT(*), 2) AS upgrade_rate_pct,
    ROUND(100.0 * SUM(pb.event = 'downgrade') / COUNT(*), 2) AS downgrade_rate_pct,
    ROUND(100.0 * SUM(pb.event = 'churn') / COUNT(*), 2) AS churn_rate_pct
FROM plan_before pb
JOIN learners l ON l.learner_id = pb.learner_id
WHERE pb.event IN ('renewal', 'upgrade', 'downgrade', 'churn')
GROUP BY l.segment
ORDER BY churn_rate_pct DESC;

-- 2c. Same, broken out by segment x plan-at-start
WITH plan_before AS (
    SELECT
        learner_id, month, event, discount_pct,
        COALESCE(LAG(plan) OVER (PARTITION BY learner_id ORDER BY month), 'Free') AS plan_at_start
    FROM engagement_panel
)
SELECT
    l.segment, pb.plan_at_start AS plan,
    COUNT(*) AS paid_months_at_risk,
    ROUND(100.0 * SUM(pb.event = 'renewal') / COUNT(*), 2) AS renewal_rate_pct,
    ROUND(100.0 * SUM(pb.event = 'upgrade') / COUNT(*), 2) AS upgrade_rate_pct,
    ROUND(100.0 * SUM(pb.event = 'churn') / COUNT(*), 2) AS churn_rate_pct
FROM plan_before pb
JOIN learners l ON l.learner_id = pb.learner_id
WHERE pb.event IN ('renewal', 'upgrade', 'downgrade', 'churn') AND pb.plan_at_start != 'Free'
GROUP BY l.segment, pb.plan_at_start
HAVING COUNT(*) >= 5
ORDER BY l.segment, FIELD(pb.plan_at_start, 'Basic', 'Pro', 'Premium');

-- 2d. Price-sensitivity / elasticity PROXY recovered from historical plan-switch
-- behavior: compare churn rate in discounted vs. non-discounted paid-months, by segment.
-- A larger relative_churn_reduction_pct = a segment that responds MORE to a discount =
-- higher empirical price sensitivity -- this is the analysis-side proxy; compare its
-- segment ordering against the true generation-time price_sensitivity_mult documented in
-- README.md to see how well it's recovered.
WITH plan_before AS (
    SELECT
        learner_id, month, event, discount_pct,
        COALESCE(LAG(plan) OVER (PARTITION BY learner_id ORDER BY month), 'Free') AS plan_at_start
    FROM engagement_panel
)
SELECT
    l.segment,
    SUM(pb.discount_pct = 0) AS months_no_discount,
    SUM(pb.discount_pct > 0) AS months_with_discount,
    ROUND(100.0 * SUM(CASE WHEN pb.discount_pct = 0 AND pb.event = 'churn' THEN 1 ELSE 0 END) / NULLIF(SUM(pb.discount_pct = 0), 0), 2) AS churn_rate_no_discount_pct,
    ROUND(100.0 * SUM(CASE WHEN pb.discount_pct > 0 AND pb.event = 'churn' THEN 1 ELSE 0 END) / NULLIF(SUM(pb.discount_pct > 0), 0), 2) AS churn_rate_with_discount_pct,
    ROUND(
        100.0 * (
            (SUM(CASE WHEN pb.discount_pct = 0 AND pb.event = 'churn' THEN 1 ELSE 0 END) / NULLIF(SUM(pb.discount_pct = 0), 0))
            - (SUM(CASE WHEN pb.discount_pct > 0 AND pb.event = 'churn' THEN 1 ELSE 0 END) / NULLIF(SUM(pb.discount_pct > 0), 0))
        ) / NULLIF((SUM(CASE WHEN pb.discount_pct = 0 AND pb.event = 'churn' THEN 1 ELSE 0 END) / NULLIF(SUM(pb.discount_pct = 0), 0)), 0)
    , 1) AS relative_churn_reduction_pct_when_discounted
FROM plan_before pb
JOIN learners l ON l.learner_id = pb.learner_id
WHERE pb.event IN ('renewal', 'upgrade', 'downgrade', 'churn')
GROUP BY l.segment
ORDER BY relative_churn_reduction_pct_when_discounted DESC;
