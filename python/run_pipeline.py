"""
End-to-end pipeline: generate synthetic data -> train churn model -> apply rules engine
-> simulate held-out intervention comparison. Run from the project root:
    python python/run_pipeline.py
"""
import pandas as pd

import generate_data as G
import ml_churn_model as M
import rules_engine as R
import simulate_intervention as S


def main():
    print("== 1. generate synthetic learners + 12-month engagement panel ==")
    learners = G.make_learners()
    panel = G.simulate_panel(learners)
    learners.to_csv("data/processed/learners.csv", index=False)
    panel.to_csv("data/processed/engagement_panel.csv", index=False)
    print(f"  {len(learners):,} learners, {len(panel):,} panel rows")

    print("== 2. train churn-risk model (optional AI layer) ==")
    churn_scores, auc = M.train_and_score(panel)
    pd.Series(churn_scores, name="churn_risk_score").rename_axis("learner_id").reset_index().to_csv(
        "data/processed/churn_risk_scores.csv", index=False
    )
    print(f"  held-out test AUC: {auc:.3f}, scored {len(churn_scores)} paid learners")

    print("== 3. apply the 24-rule recommendation engine to all learners ==")
    recs = R.recommend_for_all(learners, panel, churn_scores)
    recs.to_csv("data/processed/recommendations.csv", index=False)
    print(f"  {recs['action'].value_counts().to_dict()}")

    print("== 4. simulate held-out cohort: framework vs. baseline ==")
    import subprocess, sys
    subprocess.run([sys.executable, "python/simulate_intervention.py"], check=True)

    print("\nDONE.")


if __name__ == "__main__":
    main()
