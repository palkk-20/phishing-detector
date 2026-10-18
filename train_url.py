"""Train and evaluate the URL phishing models.

Usage
-----
    python train_url.py                      # download PhiUSIIL from UCI (first run)
    python train_url.py --csv my_urls.csv    # use your own CSV with columns: url,label
                                             # (label: 1 = phishing, 0 = legitimate)
    python train_url.py --refresh            # ignore the cached copy and re-download
    python train_url.py --no-augment         # train on the raw dataset (shows the dataset bias)

Outputs (in ./artifacts): url_models.joblib, url_metrics.json,
url_meta.json, url_false_negatives.csv
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.augment import augment_legitimate
from src.evaluation import compute_metrics, mean_latency_ms
from src.features import FEATURE_NAMES, extract_url_features

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)
CACHE = DATA / "phiusiil_urls.csv"


def download_phiusiil() -> pd.DataFrame:
    """Download PhiUSIIL (UCI id 967) and return columns: url, label."""
    from ucimlrepo import fetch_ucirepo

    print("Downloading the PhiUSIIL phishing URL dataset from UCI...")
    dataset = fetch_ucirepo(id=967)
    X_original = dataset.data.features
    y_original = dataset.data.targets

    url_column = next(
        (c for c in X_original.columns if c.strip().lower() == "url"), None
    )
    if url_column is None:
        raise ValueError(
            f"URL column not found. Available columns: {X_original.columns.tolist()}"
        )
    if y_original.shape[1] != 1:
        raise ValueError(
            f"Expected exactly one target column. Found: {y_original.columns.tolist()}"
        )
    target_column = y_original.columns[0]

    df = pd.DataFrame({
        "url": X_original[url_column],
        "original_label": y_original[target_column],
    })
    df = df.dropna(subset=["url", "original_label"]).copy()
    df["url"] = df["url"].astype(str).str.strip()
    df = df[df["url"] != ""]
    df["original_label"] = pd.to_numeric(df["original_label"], errors="raise")

    # PhiUSIIL official convention: 0 = phishing, 1 = legitimate.
    actual = set(df["original_label"].unique())
    if actual != {0, 1}:
        raise ValueError(f"Unexpected target labels: {actual}. Expected 0 and 1.")

    # Our convention: 1 = phishing, 0 = legitimate.
    df["label"] = df["original_label"].map({0: 1, 1: 0})
    return df[["url", "label"]]


def load_dataset(args) -> pd.DataFrame:
    if args.csv:
        print(f"Loading URLs from {args.csv}")
        df = pd.read_csv(args.csv)
        if not {"url", "label"}.issubset(df.columns):
            raise ValueError("CSV must have columns: url,label (1=phishing, 0=legitimate)")
    elif CACHE.exists() and not args.refresh:
        print(f"Using cached dataset {CACHE} (use --refresh to re-download)")
        df = pd.read_csv(CACHE)
    else:
        df = download_phiusiil()
        df.to_csv(CACHE, index=False)
        print(f"Cached dataset to {CACHE}")

    df = df.dropna(subset=["url", "label"]).copy()
    df["url"] = df["url"].astype(str).str.strip()
    df = df[df["url"] != ""]
    df["label"] = df["label"].astype(int)
    if set(df["label"].unique()) != {0, 1}:
        raise ValueError("Labels must contain both 0 and 1.")
    # Exact duplicate URLs are removed BEFORE splitting to avoid leakage.
    return df.drop_duplicates(subset=["url"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="CSV with columns url,label (1=phishing)")
    parser.add_argument("--refresh", action="store_true", help="re-download dataset")
    parser.add_argument(
        "--no-augment", action="store_true",
        help="do not augment the legitimate class (reproduces the dataset bias)",
    )
    args = parser.parse_args()

    df = load_dataset(args)

    augmented = not args.no_augment
    if augmented:
        # PhiUSIIL's legitimate URLs are ALL 'https://www.<domain>' with no path.
        # Replace each by one realistic variant so the model cannot use that artefact.
        legit = df["label"] == 0
        before = df.loc[legit, "url"].head(3).tolist()
        df.loc[legit, "url"] = augment_legitimate(df.loc[legit, "url"], seed=42)
        df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
        print("Augmented the legitimate class (synthetic http/https, www, subdomain, path, query variants).")
        print("  example before:", before)
        print("  example after: ", df.loc[df["label"] == 0, "url"].head(3).tolist())
    else:
        print("WARNING: --no-augment: legitimate URLs keep the raw dataset form.")

    print(f"Unique URLs: {len(df):,}")
    print("\nClass distribution (1 = phishing, 0 = legitimate):")
    print(df["label"].value_counts().sort_index())

    print("\nExtracting URL features (this may take a minute)...")
    X = pd.DataFrame(
        [extract_url_features(u) for u in df["url"]], columns=FEATURE_NAMES
    )
    y = df["label"].astype(int)

    # Split FIRST; the scaler inside the pipeline is fitted on training data only.
    X_train, X_test, y_train, y_test, url_train, url_test = train_test_split(
        X, y, df["url"], test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1,
            min_samples_leaf=10,
        ),
    }

    all_metrics = {}

    # Simple baseline: always predict the majority class.
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    all_metrics["Majority-class baseline"] = compute_metrics(
        y_test, dummy.predict(X_test), dummy.predict_proba(X_test)[:, 1]
    )

    def single_predict(model):
        def _fn(url):
            row = pd.DataFrame([extract_url_features(url)], columns=FEATURE_NAMES)
            return model.predict_proba(row)
        return _fn

    probs = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        probs[name] = proba

        m = compute_metrics(y_test, pred, proba)
        m["latency_ms_single"] = mean_latency_ms(
            single_predict(model), url_test.tolist()
        )
        all_metrics[name] = m

        print(f"Accuracy:  {m['accuracy']:.3f}")
        print(f"Precision: {m['precision']:.3f}")
        print(f"Recall:    {m['recall']:.3f}")
        print(f"F1-score:  {m['f1']:.3f}")
        print(f"False-negative rate (phishing missed): {m['false_negative_rate']:.3f}")
        print(f"Latency per URL (features + predict): {m['latency_ms_single']:.1f} ms")
        print("Confusion matrix (rows=true, cols=predicted, labels [0=legit, 1=phishing]):")
        print(m["confusion_matrix"])

    # ---- Metadata used by the app for explanations ----------------------
    rf = models["Random Forest"]
    lr = models["Logistic Regression"].steps[-1][1]
    meta = {
        "feature_names": FEATURE_NAMES,
        # typical value of each feature among LEGITIMATE training URLs
        "legit_baseline": {
            k: float(v) for k, v in X_train[y_train == 0].median().items()
        },
        "rf_importance": dict(zip(FEATURE_NAMES, map(float, rf.feature_importances_))),
        "lr_standardised_coef": dict(zip(FEATURE_NAMES, map(float, lr.coef_[0]))),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "augmented_legitimate": bool(augmented),
        "n_phishing_total": int(y.sum()),
        "n_legit_total": int((1 - y).sum()),
    }
    (ARTIFACTS / "url_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ---- Phishing URLs the Random Forest called legitimate ---------------
    fn_mask = (y_test.values == 1) & (probs["Random Forest"] < 0.5)
    fn_df = pd.DataFrame({
        "url": url_test[fn_mask].values,
        "rf_phishing_score": probs["Random Forest"][fn_mask],
        "lr_phishing_score": probs["Logistic Regression"][fn_mask],
    }).sort_values("rf_phishing_score").head(200)
    fn_df.to_csv(ARTIFACTS / "url_false_negatives.csv", index=False)

    # ---- Real-world sanity check on hand-picked URLs (tests/sanity_urls.csv) --
    sanity_path = ROOT / "tests" / "sanity_urls.csv"
    if sanity_path.exists():
        sanity = pd.read_csv(sanity_path)
        Xs = pd.DataFrame(
            [extract_url_features(u) for u in sanity["url"]], columns=FEATURE_NAMES
        )
        for name, model in models.items():
            sanity[f"{name} score"] = model.predict_proba(Xs)[:, 1].round(3)
            ok = ((sanity[f"{name} score"] >= 0.5).astype(int) == sanity["label"])
            print(f"Sanity check ({len(sanity)} hand-picked URLs) - {name}: "
                  f"{ok.sum()}/{len(sanity)} correct")
        sanity.to_csv(ARTIFACTS / "url_sanity.csv", index=False)

    joblib.dump(models, ARTIFACTS / "url_models.joblib", compress=3)
    (ARTIFACTS / "url_metrics.json").write_text(
        json.dumps(all_metrics, indent=2), encoding="utf-8"
    )

    print("\nTraining complete!")
    print(f"Saved to: {ARTIFACTS}")
    print("Reminder: test-set metrics do not guarantee performance on new,")
    print("previously unseen phishing tactics.")


if __name__ == "__main__":
    main()
