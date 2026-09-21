"""
ADDITIVE ML model #2: a segment-inference classifier.

In this project, `segment` is a ground-truth label assigned at data-generation time, and
the rules engine always has it available -- so this model is NOT wired into
rules_engine.py or the held-out intervention comparison (those numbers, and every number
in README.md's "Results" section, are unaffected by this file).

What it demonstrates instead: in a real product, a brand-new or lightly-tracked learner's
"segment" often ISN'T known upfront -- it has to be inferred from early behavior. This
model answers "how well can behavior alone recover the true segment?", using only
observable signals (engagement, tenure, geography, course category, plan) and never the
segment label itself as a feature. A multi-class Random Forest is used here (vs. the
binary logistic regressions elsewhere in this project) for genuine method diversity, and
because segment separation is not obviously linear.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

FEATURES_NUMERIC = ["logins_this_month", "course_completion_pct", "feature_usage_score", "tenure_months_on_plan"]
FEATURES_CATEGORICAL = ["geography", "course_category", "plan"]


def build_feature_table(learners: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """One row per learner: their latest panel snapshot + static learner fields."""
    last = panel.sort_values("month").groupby("learner_id").tail(1)
    df = learners.merge(last, on="learner_id")
    return df


def _fit_and_score(X, y, seed):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, output_dict=True)
    majority_class = y_train.value_counts().idxmax()
    baseline_acc = (y_test == majority_class).mean()
    feature_importance = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)

    return {
        "n_train": len(X_train), "n_test": len(X_test),
        "accuracy": acc, "macro_f1": macro_f1,
        "majority_class_baseline_accuracy": baseline_acc,
        "per_class_report": report,
        "top_features": feature_importance.head(8),
    }


def train_and_evaluate(learners: pd.DataFrame, panel: pd.DataFrame, seed=13) -> dict:
    """Two variants, reported honestly side by side:
    - 'full': numeric behavior + geography + course_category. course_category is
      deterministically 1:1 with segment at generation time (see generate_data.py's
      SEGMENTS dict) for 4 of 6 segments, so this variant's accuracy mostly reflects the
      model rediscovering that lookup, not genuine behavioral inference.
    - 'behavior_only': numeric engagement signals ONLY (logins, completion, feature
      usage, tenure) -- no category, no geography. This is the honest answer to "how
      much does pure behavior alone reveal about segment," which is what a real
      cold-start scenario (segment truly unknown) would have to rely on.
    """
    df = build_feature_table(learners, panel)
    y = df["segment"]

    X_full = pd.get_dummies(df[FEATURES_NUMERIC + FEATURES_CATEGORICAL], columns=FEATURES_CATEGORICAL)
    X_behavior_only = df[FEATURES_NUMERIC]

    full = _fit_and_score(X_full, y, seed)
    behavior_only = _fit_and_score(X_behavior_only, y, seed)
    return {"full": full, "behavior_only": behavior_only}


if __name__ == "__main__":
    learners = pd.read_csv("data/processed/learners.csv")
    panel = pd.read_csv("data/processed/engagement_panel.csv")
    results = train_and_evaluate(learners, panel)

    summary_rows = []
    for variant, result in results.items():
        print(f"\n=== {variant} ===")
        print(f"train rows: {result['n_train']}, test rows: {result['n_test']}")
        print(f"accuracy: {result['accuracy']:.3f}  (majority-class baseline: {result['majority_class_baseline_accuracy']:.3f})")
        print(f"macro F1: {result['macro_f1']:.3f}")
        print("top features by importance:")
        print(result["top_features"].to_string())
        summary_rows.append({
            "variant": variant, "n_train": result["n_train"], "n_test": result["n_test"],
            "accuracy": result["accuracy"], "macro_f1": result["macro_f1"],
            "majority_class_baseline_accuracy": result["majority_class_baseline_accuracy"],
        })

    pd.DataFrame(summary_rows).to_csv("output/tables/ml_segment_model_summary.csv", index=False)

    per_class_rows = []
    for variant, result in results.items():
        for seg, m in result["per_class_report"].items():
            if seg in ("accuracy", "macro avg", "weighted avg"):
                continue
            per_class_rows.append({"variant": variant, "segment": seg, "precision": m["precision"],
                                     "recall": m["recall"], "f1": m["f1-score"], "support": m["support"]})
    pd.DataFrame(per_class_rows).to_csv("output/tables/ml_segment_model_per_class.csv", index=False)
