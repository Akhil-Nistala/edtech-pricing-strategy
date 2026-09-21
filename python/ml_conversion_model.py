"""
ADDITIVE ML model #3: a free-to-paid conversion-propensity classifier.

Same status as ml_segment_model.py: this is NOT wired into rules_engine.py or the
held-out intervention comparison, so none of README.md's existing "Results" numbers
(3.81% conversion, the 24-rule engine, the framework-vs-baseline comparison) are affected
by this file. It exists to answer a different, complementary question: for a FREE learner
in a given month, how well does that month's observable behavior predict whether they
convert to paid THIS month?

Training examples: every learner-month where the learner was on the Free plan and a
following month exists; label = 1 if the NEXT month's event is "conversion". Same
methodology as ml_churn_model.py's churn-risk classifier (logistic regression,
class_weight="balanced", held-out AUC), applied to the mirror-image event.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = ["logins_this_month", "course_completion_pct", "feature_usage_score", "month"]


def build_training_set(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lid, g in panel.groupby("learner_id"):
        g = g.sort_values("month").reset_index(drop=True)
        for i in range(len(g) - 1):
            if g.loc[i, "plan"] != "Free":
                continue
            next_event = g.loc[i + 1, "event"]
            row = g.loc[i, FEATURES].to_dict()
            row["learner_id"] = lid
            row["label_converts_next_month"] = 1 if next_event == "conversion" else 0
            rows.append(row)
    return pd.DataFrame(rows)


def train_and_score(panel: pd.DataFrame, seed=17) -> tuple[dict, float, int]:
    train_df = build_training_set(panel)
    X, y = train_df[FEATURES], train_df["label_converts_next_month"]

    if y.sum() < 10:
        raise RuntimeError(f"only {y.sum()} positive (conversion) examples -- insufficient to train responsibly")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit(scaler.transform(X_train), y_train)

    test_auc = roc_auc_score(y_test, clf.predict_proba(scaler.transform(X_test))[:, 1])

    scores = {}
    for lid, g in panel.groupby("learner_id"):
        g = g.sort_values("month")
        last_free = g[g["plan"] == "Free"]
        if len(last_free) == 0:
            continue
        last_row = last_free.iloc[[-1]][FEATURES]
        scores[lid] = float(clf.predict_proba(scaler.transform(last_row))[0, 1])

    return scores, test_auc, int(y.sum())


if __name__ == "__main__":
    panel = pd.read_csv("data/processed/engagement_panel.csv")
    train_df = build_training_set(panel)
    scores, auc, n_positive = train_and_score(panel)
    print(f"trained on {len(train_df):,} free-state learner-month rows ({n_positive} positive/converted)")
    print(f"held-out test AUC: {auc:.3f}")
    print(f"scored {len(scores):,} currently-free learners")

    pd.Series(scores, name="conversion_propensity_score").rename_axis("learner_id").reset_index().to_csv(
        "data/processed/conversion_propensity_scores.csv", index=False
    )
    pd.DataFrame([{
        "n_training_rows": len(train_df), "n_positive": n_positive,
        "test_auc": auc, "n_scored": len(scores),
    }]).to_csv("output/tables/ml_conversion_model_summary.csv", index=False)
