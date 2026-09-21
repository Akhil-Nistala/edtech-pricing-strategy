"""
Optional "AI beyond rules" layer: a simple, transparent logistic regression predicting
each paid learner's probability of churning next month, trained on the synthetic
engagement panel. This is explicitly a SUPPLEMENT to the rules engine, not a replacement
for it -- only 2 of the 24 rules (R09, R10) consume this score, and both remain
well-defined (they just won't fire) if this model is removed entirely. This mirrors the
brief's own suggestion: "optionally wrapped with a simple ML classifier ... if you want
genuine AI beyond rules."

Training examples: for every learner-month where the learner was on a paid plan AND a
following month exists, features = that month's engagement snapshot, label = whether the
FOLLOWING month's event was "churn". Because the synthetic population only has ~330 total
paid-state months (see README > Data Assumptions on the resulting sample-size caveat),
this is a small-N logistic regression demonstrating the mechanism, not a production-grade
model -- stated plainly rather than overclaiming its reliability.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "logins_this_month", "course_completion_pct", "feature_usage_score",
    "tenure_months_on_plan", "discount_pct",
]


def build_training_set(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lid, g in panel.groupby("learner_id"):
        g = g.sort_values("month").reset_index(drop=True)
        for i in range(len(g) - 1):
            if g.loc[i, "plan"] == "Free":
                continue
            next_event = g.loc[i + 1, "event"]
            if next_event not in ("renewal", "upgrade", "downgrade", "churn"):
                continue
            row = g.loc[i, FEATURES].to_dict()
            row["learner_id"] = lid
            row["month"] = g.loc[i, "month"]
            row["label_churn_next_month"] = 1 if next_event == "churn" else 0
            rows.append(row)
    return pd.DataFrame(rows)


def train_and_score(panel: pd.DataFrame, seed=5) -> tuple[dict, float]:
    train_df = build_training_set(panel)
    X, y = train_df[FEATURES], train_df["label_churn_next_month"]

    if y.nunique() < 2 or len(train_df) < 30:
        # Too little signal to fit responsibly -- fail loudly rather than return a fake model.
        raise RuntimeError(f"only {len(train_df)} training rows, {y.nunique()} label classes -- insufficient to train")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit(scaler.transform(X_train), y_train)

    test_auc = roc_auc_score(y_test, clf.predict_proba(scaler.transform(X_test))[:, 1])

    # Score every learner's MOST RECENT paid-state month (their current standing) for use
    # by rules_engine.py.
    scores = {}
    for lid, g in panel.groupby("learner_id"):
        g = g.sort_values("month")
        last_paid = g[g["plan"] != "Free"]
        if len(last_paid) == 0:
            continue
        last_row = last_paid.iloc[[-1]][FEATURES]
        scores[lid] = float(clf.predict_proba(scaler.transform(last_row))[0, 1])

    return scores, test_auc


if __name__ == "__main__":
    panel = pd.read_csv("data/processed/engagement_panel.csv")
    scores, auc = train_and_score(panel)
    print(f"trained on {len(build_training_set(panel))} learner-month rows")
    print(f"held-out test AUC: {auc:.3f}")
    print(f"scored {len(scores)} currently-paid learners")
    pd.Series(scores, name="churn_risk_score").rename_axis("learner_id").reset_index().to_csv(
        "data/processed/churn_risk_scores.csv", index=False
    )
