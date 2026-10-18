"""Train and evaluate the email-text phishing models (TF-IDF).

Usage
-----
    python train_email.py --demo                 # tiny hand-written demo set (pipeline test only!)
    python train_email.py --csv data/emails.csv  # your real dataset

The CSV needs one text column and one label column. Common names are
detected automatically (e.g. "Email Text" / "Email Type"), or pass
--text-col and --label-col.

Label handling
--------------
Text labels containing phish/spam/fraud/scam -> 1 (phishing)
Text labels containing safe/ham/legit/benign -> 0 (legitimate)
Numeric labels are assumed 1 = phishing. If your dataset uses the
opposite convention, add --flip-labels.

Outputs (in ./artifacts): email_models.joblib, email_metrics.json,
email_meta.json, email_false_negatives.csv
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.evaluation import compute_metrics, mean_latency_ms
from src.text_utils import clean_text

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
DEMO_CSV = ROOT / "data" / "sample_emails.csv"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

TEXT_CANDIDATES = ["text", "email text", "email_text", "body", "message", "content", "email"]
LABEL_CANDIDATES = ["label", "email type", "email_type", "class", "type", "target", "spam"]
POS_WORDS = ("phish", "spam", "fraud", "scam", "malicious")
NEG_WORDS = ("safe", "ham", "legit", "benign", "normal")


def find_column(df, candidates, explicit):
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Column '{explicit}' not found. Columns: {df.columns.tolist()}")
        return explicit
    lower = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    raise ValueError(
        f"Could not detect the column. Columns are: {df.columns.tolist()}. "
        "Use --text-col / --label-col."
    )


def map_labels(series, flip):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        print("NOTE: numeric labels detected - assuming 1 = phishing, 0 = legitimate.")
        labels = numeric.astype(int)
    else:
        def one(v):
            s = str(v).strip().lower()
            if any(w in s for w in POS_WORDS):
                return 1
            if any(w in s for w in NEG_WORDS):
                return 0
            raise ValueError(f"Cannot interpret label value: {v!r}")
        labels = series.map(one)
    if flip:
        labels = 1 - labels
    return labels.astype(int)


def make_pipeline(classifier, min_df):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=clean_text,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=min_df,
            max_features=50000,
            sublinear_tf=True,
        )),
        ("clf", classifier),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="path to the email dataset CSV")
    parser.add_argument("--demo", action="store_true", help="use the tiny demo dataset")
    parser.add_argument("--text-col")
    parser.add_argument("--label-col")
    parser.add_argument("--flip-labels", action="store_true")
    args = parser.parse_args()

    if args.demo:
        path = DEMO_CSV
        print("WARNING: demo mode - only ~40 hand-written emails. Use it to test the")
        print("pipeline and the app. Do NOT report these metrics in your project.\n")
    elif args.csv:
        path = Path(args.csv)
    else:
        raise SystemExit("Provide --csv path/to/emails.csv (or --demo for a quick test).")

    df = pd.read_csv(path)
    text_col = find_column(df, TEXT_CANDIDATES, args.text_col)
    label_col = find_column(df, LABEL_CANDIDATES, args.label_col)
    print(f"Using text column '{text_col}' and label column '{label_col}'")

    df = df[[text_col, label_col]].dropna().copy()
    df.columns = ["text", "raw_label"]
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    df["label"] = map_labels(df["raw_label"], args.flip_labels)

    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"Removed {before - len(df)} duplicate emails; {len(df):,} remain.")
    print("\nClass distribution (1 = phishing, 0 = legitimate):")
    print(df["label"].value_counts().sort_index())

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.20, random_state=42, stratify=df["label"]
    )

    min_df = 1 if len(df) < 500 else 2
    models = {
        "Logistic Regression": make_pipeline(
            LogisticRegression(max_iter=2000, class_weight="balanced"), min_df
        ),
        "Naive Bayes": make_pipeline(MultinomialNB(alpha=0.1), min_df),
    }

    all_metrics = {}
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    all_metrics["Majority-class baseline"] = compute_metrics(
        y_test, dummy.predict(X_test), dummy.predict_proba(X_test)[:, 1]
    )

    probs = {}
    for name, model in models.items():
        print(f"\nTraining {name} (TF-IDF fitted on training data only)...")
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        probs[name] = proba

        m = compute_metrics(y_test, pred, proba)
        m["latency_ms_single"] = mean_latency_ms(
            lambda t, mdl=model: mdl.predict_proba([t]), X_test.tolist()
        )
        all_metrics[name] = m

        print(f"Accuracy:  {m['accuracy']:.3f}")
        print(f"Precision: {m['precision']:.3f}")
        print(f"Recall:    {m['recall']:.3f}")
        print(f"F1-score:  {m['f1']:.3f}")
        print(f"False-negative rate (phishing missed): {m['false_negative_rate']:.3f}")
        print(f"Latency per email: {m['latency_ms_single']:.1f} ms")
        print("Confusion matrix (rows=true, cols=predicted, labels [0=legit, 1=phishing]):")
        print(m["confusion_matrix"])

    # Phishing emails predicted as legitimate by Logistic Regression
    lr_p = probs["Logistic Regression"]
    mask = (y_test.values == 1) & (lr_p < 0.5)
    fn_df = pd.DataFrame({
        "text_preview": X_test[mask].str.slice(0, 300).values,
        "lr_phishing_score": lr_p[mask],
        "nb_phishing_score": probs["Naive Bayes"][mask],
    }).sort_values("lr_phishing_score").head(100)
    fn_df.to_csv(ARTIFACTS / "email_false_negatives.csv", index=False)

    meta = {
        "source_file": str(path.name),
        "demo_mode": bool(args.demo),
        "n_emails": int(len(df)),
        "n_phishing": int(df["label"].sum()),
        "n_legit": int((1 - df["label"]).sum()),
        "n_test": int(len(y_test)),
    }
    (ARTIFACTS / "email_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    joblib.dump(models, ARTIFACTS / "email_models.joblib")
    (ARTIFACTS / "email_metrics.json").write_text(
        json.dumps(all_metrics, indent=2), encoding="utf-8"
    )

    print("\nTraining complete!")
    print(f"Saved to: {ARTIFACTS}")
    if args.demo:
        print("Remember: this was DEMO data - metrics are not meaningful.")


if __name__ == "__main__":
    main()
