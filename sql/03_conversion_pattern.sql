-- ============================================================================
-- Conversion pattern analysis -- SYNTHETIC data (see README > Data Assumptions).
-- Freemium-to-paid conversion by segment, and time-to-convert distribution.
-- ============================================================================

USE edtech_pricing_strategy;

-- 3a. Freemium-to-paid conversion rate by segment (cumulative over the 12-month panel)
SELECT
    l.segment,
    COUNT(DISTINCT l.learner_id) AS learners,
    COUNT(DISTINCT CASE WHEN ep.event = 'conversion' THEN ep.learner_id END) AS converters,
    ROUND(100.0 * COUNT(DISTINCT CASE WHEN ep.event = 'conversion' THEN ep.learner_id END)
          / COUNT(DISTINCT l.learner_id), 2) AS conversion_rate_pct
FROM learners l
LEFT JOIN engagement_panel ep ON ep.learner_id = l.learner_id
GROUP BY l.segment
ORDER BY conversion_rate_pct DESC;

-- 3b. Overall conversion rate (cite against the ~2-5% general / ~2.6% EdTech benchmark)
SELECT
    COUNT(DISTINCT l.learner_id) AS total_learners,
    COUNT(DISTINCT CASE WHEN ep.event = 'conversion' THEN ep.learner_id END) AS converters,
    ROUND(100.0 * COUNT(DISTINCT CASE WHEN ep.event = 'conversion' THEN ep.learner_id END)
          / COUNT(DISTINCT l.learner_id), 2) AS conversion_rate_pct
FROM learners l
LEFT JOIN engagement_panel ep ON ep.learner_id = l.learner_id;

-- 3c. Time-to-convert distribution (the `month` value at which conversion happened).
-- Median via a window-function rank trick, not LIMIT+OFFSET -- MySQL's LIMIT/OFFSET
-- can't take a subquery (same fix as the sibling restaurant-health-monitoring project).
WITH ranked AS (
    SELECT month, ROW_NUMBER() OVER (ORDER BY month) AS rn, COUNT(*) OVER () AS cnt
    FROM engagement_panel WHERE event = 'conversion'
)
SELECT
    (SELECT MIN(month) FROM engagement_panel WHERE event = 'conversion') AS min_month,
    (SELECT MAX(month) FROM engagement_panel WHERE event = 'conversion') AS max_month,
    (SELECT ROUND(AVG(month), 2) FROM engagement_panel WHERE event = 'conversion') AS avg_month,
    (SELECT ROUND(AVG(month), 1) FROM ranked WHERE rn IN (FLOOR((cnt + 1) / 2), FLOOR((cnt + 2) / 2))) AS median_month;

-- 3d. Time-to-convert histogram (converters by the month they converted in)
SELECT month AS converted_in_month, COUNT(*) AS converters
FROM engagement_panel
WHERE event = 'conversion'
GROUP BY month
ORDER BY month;

-- 3e. Time-to-convert by segment (average month of conversion)
SELECT
    l.segment,
    COUNT(*) AS converters,
    ROUND(AVG(ep.month), 2) AS avg_month_to_convert,
    MIN(ep.month) AS fastest_month,
    MAX(ep.month) AS slowest_month
FROM engagement_panel ep
JOIN learners l ON l.learner_id = ep.learner_id
WHERE ep.event = 'conversion'
GROUP BY l.segment
ORDER BY avg_month_to_convert;
