-- ============================================================================
-- Schema for the EdTech pricing-strategy project. Target: MySQL 8.0+.
--
-- EVERY table here is 100% SYNTHETIC -- no real learner data exists anywhere in this
-- project. See README.md > "Data Assumptions" for exactly how each field was generated
-- and which real industry benchmark it's calibrated against.
-- ============================================================================

CREATE DATABASE IF NOT EXISTS edtech_pricing_strategy;
USE edtech_pricing_strategy;

SET FOREIGN_KEY_CHECKS = 0;
DROP TABLE IF EXISTS intervention_comparison;
DROP TABLE IF EXISTS heldout_sim_baseline;
DROP TABLE IF EXISTS heldout_sim_framework;
DROP TABLE IF EXISTS heldout_learners;
DROP TABLE IF EXISTS churn_risk_scores;
DROP TABLE IF EXISTS recommendations;
DROP TABLE IF EXISTS engagement_panel;
DROP TABLE IF EXISTS learners;
SET FOREIGN_KEY_CHECKS = 1;

-- SYNTHETIC: one row per simulated learner.
CREATE TABLE learners (
    learner_id          VARCHAR(16) PRIMARY KEY,
    segment             VARCHAR(32),      -- one of 6 named segments, see README
    geography           VARCHAR(16),      -- Metro / Tier-2 / Tier-3
    course_category     VARCHAR(32),
    signup_day_offset   SMALLINT
) ENGINE=InnoDB;
CREATE INDEX idx_learners_segment ON learners (segment);

-- SYNTHETIC: one row per learner per month (12 months), the full engagement + plan +
-- pricing + lifecycle-event panel.
CREATE TABLE engagement_panel (
    learner_id              VARCHAR(16),
    month                   TINYINT,
    plan                    VARCHAR(8),   -- Free / Basic / Pro / Premium (plan AFTER this month's event)
    event                   VARCHAR(12),  -- none / conversion / renewal / upgrade / downgrade / churn
    list_price              INT,
    price_paid              DECIMAL(8,2),
    discount_pct            DECIMAL(4,3),
    logins_this_month       DECIMAL(6,2),
    course_completion_pct   DECIMAL(5,2),
    feature_usage_score     TINYINT,
    tenure_months_on_plan   TINYINT,
    PRIMARY KEY (learner_id, month),
    CONSTRAINT fk_panel_learner FOREIGN KEY (learner_id) REFERENCES learners(learner_id)
) ENGINE=InnoDB;
CREATE INDEX idx_panel_plan ON engagement_panel (plan);
CREATE INDEX idx_panel_event ON engagement_panel (event);

-- Derived: the rules engine's recommendation for each learner, evaluated on their
-- month-12 (latest) state.
CREATE TABLE recommendations (
    learner_id     VARCHAR(16) PRIMARY KEY,
    rule_id        VARCHAR(4),
    action         VARCHAR(32),
    target_plan    VARCHAR(8),
    discount_pct   DECIMAL(4,3),
    rationale      VARCHAR(500),
    CONSTRAINT fk_recs_learner FOREIGN KEY (learner_id) REFERENCES learners(learner_id)
) ENGINE=InnoDB;

-- Derived: the optional ML churn-risk score (one of the 14 rule-engine input parameters).
CREATE TABLE churn_risk_scores (
    learner_id        VARCHAR(16) PRIMARY KEY,
    churn_risk_score  DECIMAL(6,5),
    CONSTRAINT fk_scores_learner FOREIGN KEY (learner_id) REFERENCES learners(learner_id)
) ENGINE=InnoDB;

-- SYNTHETIC: the held-out cohort (a fresh, never-used-for-training population) used to
-- compare the rules-engine framework against a baseline offer strategy.
CREATE TABLE heldout_learners (
    learner_id          VARCHAR(16) PRIMARY KEY,
    segment             VARCHAR(32),
    geography           VARCHAR(16),
    course_category     VARCHAR(32),
    signup_day_offset   SMALLINT
) ENGINE=InnoDB;

CREATE TABLE heldout_sim_framework (
    row_id            INT AUTO_INCREMENT PRIMARY KEY,
    learner_id        VARCHAR(16),
    arm               VARCHAR(16),
    starting_state    VARCHAR(8),
    action            VARCHAR(32),
    discount_pct      DECIMAL(4,3),
    converted         VARCHAR(8),
    renewed           VARCHAR(8),
    upgraded          VARCHAR(8),
    churned           VARCHAR(8)
) ENGINE=InnoDB;
CREATE INDEX idx_fw_learner ON heldout_sim_framework (learner_id);

CREATE TABLE heldout_sim_baseline (
    row_id            INT AUTO_INCREMENT PRIMARY KEY,
    learner_id        VARCHAR(16),
    arm               VARCHAR(16),
    starting_state    VARCHAR(8),
    action            VARCHAR(32),
    discount_pct      DECIMAL(4,3),
    converted         VARCHAR(8),
    renewed           VARCHAR(8),
    upgraded          VARCHAR(8),
    churned           VARCHAR(8)
) ENGINE=InnoDB;
CREATE INDEX idx_bl_learner ON heldout_sim_baseline (learner_id);

-- The final metrics comparison table (also computed directly in Python; loaded here for
-- convenience/parity).
CREATE TABLE intervention_comparison (
    metric               VARCHAR(32) PRIMARY KEY,
    n_baseline           INT,
    n_framework          INT,
    baseline_rate        DECIMAL(6,2),
    framework_rate        DECIMAL(6,2),
    absolute_lift_pp       DECIMAL(6,2),
    relative_lift_pct       DECIMAL(8,2),
    z_statistic              DECIMAL(8,3),
    p_value                   DOUBLE
) ENGINE=InnoDB;
