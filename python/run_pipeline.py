"""
End-to-end pipeline: generate synthetic data -> train churn model -> apply rules engine
-> simulate held-out intervention comparison -> (additive) train segment + conversion
models. Run from the project root:
    python python/run_pipeline.py

Steps 5 and 6 (ml_segment_model, ml_conversion_model) are ADDITIVE analysis -- they do
not feed the rules engine or the held-out intervention comparison, so nothing in steps
1-4 depends on them or changes if they're skipped. See their module docstrings and
README.md > "ML Layer" for why.
"""
import pandas as pd

import generate_data as G
import ml_churn_model as M
import ml_conversion_model as CV
import ml_segment_model as SG
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

    print("== 5. train segment-inference model (additive, analysis-only) ==")
    seg_results = SG.train_and_evaluate(learners, panel)
    for variant, r in seg_results.items():
        print(f"  {variant}: accuracy={r['accuracy']:.3f} (baseline={r['majority_class_baseline_accuracy']:.3f}) macro_f1={r['macro_f1']:.3f}")
    summary_rows = [{"variant": v, "n_train": r["n_train"], "n_test": r["n_test"], "accuracy": r["accuracy"],
                      "macro_f1": r["macro_f1"], "majority_class_baseline_accuracy": r["majority_class_baseline_accuracy"]}
                     for v, r in seg_results.items()]
    pd.DataFrame(summary_rows).to_csv("output/tables/ml_segment_model_summary.csv", index=False)

    print("== 6. train conversion-propensity model (additive, analysis-only) ==")
    conv_scores, conv_auc, n_positive = CV.train_and_score(panel)
    print(f"  held-out test AUC: {conv_auc:.3f} ({n_positive} positive examples)")
    pd.Series(conv_scores, name="conversion_propensity_score").rename_axis("learner_id").reset_index().to_csv(
        "data/processed/conversion_propensity_scores.csv", index=False
    )
    pd.DataFrame([{"n_positive": n_positive, "test_auc": conv_auc, "n_scored": len(conv_scores)}]).to_csv(
        "output/tables/ml_conversion_model_summary.csv", index=False
    )

    print("\nDONE.")


if __name__ == "__main__":
    main()
