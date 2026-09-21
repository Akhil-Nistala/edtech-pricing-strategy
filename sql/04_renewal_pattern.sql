-- ============================================================================
-- Renewal pattern analysis -- SYNTHETIC data (see README > Data Assumptions).
-- Renewal rate by plan / segment / tenure, and cohort-based renewal curves.
-- ============================================================================

USE edtech_pricing_strategy;

-- NOTE on tenure_months_on_plan: this column reflects tenure AFTER that month's event is
-- applied (it resets to 0 exactly when a churn/upgrade/downgrade happens, and increments
-- exactly when a renewal happens) -- grouping by the raw column would be circular (the
-- "tenure=0" bucket would by definition contain 0% renewals, since resetting-to-0 IS what
-- a non-renewal event does). LAG() recovers the tenure a learner had ENTERING the month,
-- which is the value that could plausibly have influenced that month's renewal decision.
WITH plan_before AS (
    SELECT
        learner_id, month, event,
        COALESCE(LAG(tenure_months_on_plan) OVER (PARTITION BY learner_id ORDER BY month), 0) AS tenure_at_start,
        COALESCE(LAG(plan) OVER (PARTITION BY learner_id ORDER BY month), 'Free') AS plan_at_start
    FROM engagement_panel
)

-- 4a. Renewal rate by plan (Basic/Pro/Premium)
SELECT
    plan_at_start AS plan,
    COUNT(*) AS paid_months_at_risk,
    ROUND(100.0 * SUM(event = 'renewal') / COUNT(*), 2) AS renewal_rate_pct,
    ROUND(100.0 * SUM(event = 'churn') / COUNT(*), 2) AS churn_rate_pct
FROM plan_before
WHERE event IN ('renewal', 'upgrade', 'downgrade', 'churn') AND plan_at_start != 'Free'
GROUP BY plan_at_start
ORDER BY FIELD(plan_at_start, 'Basic', 'Pro', 'Premium');

-- 4b. Renewal rate by TENURE bucket (does loyalty grow with tenure? classic survival pattern)
WITH plan_before AS (
    SELECT
        learner_id, month, event,
        COALESCE(LAG(tenure_months_on_plan) OVER (PARTITION BY learner_id ORDER BY month), 0) AS tenure_at_start,
        COALESCE(LAG(plan) OVER (PARTITION BY learner_id ORDER BY month), 'Free') AS plan_at_start
    FROM engagement_panel
)
SELECT
    CASE
        WHEN tenure_at_start = 0 THEN '0 (first renewal decision)'
        WHEN tenure_at_start BETWEEN 1 AND 2 THEN '1-2 months'
        WHEN tenure_at_start BETWEEN 3 AND 5 THEN '3-5 months'
        ELSE '6+ months'
    END AS tenure_bucket,
    COUNT(*) AS paid_months_at_risk,
    ROUND(100.0 * SUM(event = 'renewal') / COUNT(*), 2) AS renewal_rate_pct,
    ROUND(100.0 * SUM(event = 'churn') / COUNT(*), 2) AS churn_rate_pct
FROM plan_before
WHERE event IN ('renewal', 'upgrade', 'downgrade', 'churn') AND plan_at_start != 'Free'
GROUP BY tenure_bucket
ORDER BY FIELD(tenure_bucket, '0 (first renewal decision)', '1-2 months', '3-5 months', '6+ months');

-- 4c. Cohort-based renewal (survival) curve: cohort = the month a learner converted in;
-- for each cohort, % still on a PAID plan (has not reverted to Free) N months after
-- their own conversion month.
WITH conversions AS (
    SELECT learner_id, month AS conversion_month
    FROM engagement_panel WHERE event = 'conversion'
),
post_conversion AS (
    SELECT
        c.conversion_month,
        ep.learner_id,
        ep.month - c.conversion_month AS months_since_conversion,
        ep.plan
    FROM conversions c
    JOIN engagement_panel ep ON ep.learner_id = c.learner_id AND ep.month > c.conversion_month
)
SELECT
    conversion_month AS cohort_conversion_month,
    months_since_conversion,
    COUNT(*) AS learners_in_cohort_at_this_point,
    ROUND(100.0 * SUM(plan != 'Free') / COUNT(*), 2) AS pct_still_paid
FROM post_conversion
WHERE months_since_conversion BETWEEN 1 AND 6
GROUP BY conversion_month, months_since_conversion
HAVING COUNT(*) >= 3
ORDER BY conversion_month, months_since_conversion;

-- 4d. Renewal rate by segment x tenure bucket (combined view)
WITH plan_before AS (
    SELECT
        learner_id, month, event,
        COALESCE(LAG(tenure_months_on_plan) OVER (PARTITION BY learner_id ORDER BY month), 0) AS tenure_at_start,
        COALESCE(LAG(plan) OVER (PARTITION BY learner_id ORDER BY month), 'Free') AS plan_at_start
    FROM engagement_panel
)
SELECT
    l.segment,
    CASE WHEN pb.tenure_at_start >= 3 THEN '3+ months' ELSE '0-2 months' END AS tenure_bucket,
    COUNT(*) AS paid_months_at_risk,
    ROUND(100.0 * SUM(pb.event = 'renewal') / COUNT(*), 2) AS renewal_rate_pct
FROM plan_before pb
JOIN learners l ON l.learner_id = pb.learner_id
WHERE pb.event IN ('renewal', 'upgrade', 'downgrade', 'churn') AND pb.plan_at_start != 'Free'
GROUP BY l.segment, tenure_bucket
HAVING COUNT(*) >= 5
ORDER BY l.segment, tenure_bucket;
