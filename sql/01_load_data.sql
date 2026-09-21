-- ============================================================================
-- Loads the synthetic CSVs (produced by python/run_pipeline.py) into MySQL.
--
-- Run with (from the project root, so the relative paths below resolve):
--   mysql --local-infile=1 -u root -p edtech_pricing_strategy < sql/01_load_data.sql
--
-- Requires local_infile enabled SERVER-side too (resets on every MySQL restart):
--   mysql -u root -p -e "SET GLOBAL local_infile = 1;"
--
-- LINES TERMINATED BY '\r\n' is declared explicitly from the start here (not '\n'),
-- learned the hard way on the two sibling projects in this series: pandas writes
-- Windows \r\n line endings, and declaring only '\n' either corrupts an unquoted last
-- field with a stray \r, or -- worse, when the last field is quoted and contains the
-- delimiter character -- silently drops the vast majority of rows with a generic
-- "too many columns" warning. Declaring the real terminator upfront avoids both.
-- ============================================================================

USE edtech_pricing_strategy;

LOAD DATA LOCAL INFILE 'data/processed/learners.csv'
INTO TABLE learners
FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 LINES
(learner_id, segment, geography, course_category, signup_day_offset);

LOAD DATA LOCAL INFILE 'data/processed/engagement_panel.csv'
INTO TABLE engagement_panel
FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 LINES
(learner_id, month, plan, event, list_price, price_paid, discount_pct,
 logins_this_month, course_completion_pct, feature_usage_score, tenure_months_on_plan);

LOAD DATA LOCAL INFILE 'data/processed/recommendations.csv'
INTO TABLE recommendations
FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 LINES
(learner_id, rule_id, action, @target_plan, discount_pct, rationale)
SET target_plan = NULLIF(@target_plan, '');

LOAD DATA LOCAL INFILE 'data/processed/churn_risk_scores.csv'
INTO TABLE churn_risk_scores
FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 LINES
(learner_id, churn_risk_score);

LOAD DATA LOCAL INFILE 'data/processed/heldout_learners.csv'
INTO TABLE heldout_learners
FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 LINES
(learner_id, segment, geography, course_category, signup_day_offset);

LOAD DATA LOCAL INFILE 'output/tables/heldout_sim_framework.csv'
INTO TABLE heldout_sim_framework
FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 LINES
(learner_id, arm, starting_state, action, discount_pct, @c, @rn, @up, @ch)
SET converted = NULLIF(@c, ''), renewed = NULLIF(@rn, ''), upgraded = NULLIF(@up, ''), churned = NULLIF(@ch, '');

LOAD DATA LOCAL INFILE 'output/tables/heldout_sim_baseline.csv'
INTO TABLE heldout_sim_baseline
FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 LINES
(learner_id, arm, starting_state, action, discount_pct, @c, @rn, @up, @ch)
SET converted = NULLIF(@c, ''), renewed = NULLIF(@rn, ''), upgraded = NULLIF(@up, ''), churned = NULLIF(@ch, '');

LOAD DATA LOCAL INFILE 'output/tables/intervention_comparison.csv'
INTO TABLE intervention_comparison
FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 LINES
(metric, n_baseline, n_framework, baseline_rate, framework_rate, absolute_lift_pp,
 relative_lift_pct, z_statistic, p_value);

SELECT 'learners' AS tbl, COUNT(*) AS rows_loaded FROM learners
UNION ALL SELECT 'engagement_panel', COUNT(*) FROM engagement_panel
UNION ALL SELECT 'recommendations', COUNT(*) FROM recommendations
UNION ALL SELECT 'churn_risk_scores', COUNT(*) FROM churn_risk_scores
UNION ALL SELECT 'heldout_learners', COUNT(*) FROM heldout_learners
UNION ALL SELECT 'heldout_sim_framework', COUNT(*) FROM heldout_sim_framework
UNION ALL SELECT 'heldout_sim_baseline', COUNT(*) FROM heldout_sim_baseline
UNION ALL SELECT 'intervention_comparison', COUNT(*) FROM intervention_comparison;
