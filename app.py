"""Explainable AI Phishing Detector - Streamlit app.

Run with:  streamlit run app.py
The app only analyses text. It NEVER opens, fetches or visits a URL.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.decision import LEGIT, PHISHING, UNCERTAIN, decide
from src.explain import (
    email_risk_indicators,
    email_term_contributions,
    occlusion_explanation,
    url_risk_indicators,
)
from src.features import FEATURE_NAMES, extract_url_feature_dict

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

st.set_page_config(page_title="Explainable Phishing Detector", page_icon="🛡️", layout="wide")


# ------------------------------------------------------------- loading ----
@st.cache_resource
def load_models(kind: str):
    path = ARTIFACTS / f"{kind}_models.joblib"
    return joblib.load(path) if path.exists() else None


@st.cache_data
def load_json(name: str):
    path = ARTIFACTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@st.cache_data
def load_csv(name: str):
    path = ARTIFACTS / name
    return pd.read_csv(path) if path.exists() else None


url_models = load_models("url")
email_models = load_models("email")
url_meta = load_json("url_meta.json")
email_meta = load_json("email_meta.json")

SEVERITY_ICON = {"high": "🔴", "medium": "🟠", "low": "🟡", "info": "ℹ️"}


# ------------------------------------------------------------- helpers ----
def show_verdict(verdict, p, reasons):
    if verdict == PHISHING:
        st.error(f"### 🚨 {verdict}\nModel score (phishing): **{p:.2f}**")
    elif verdict == LEGIT:
        st.success(f"### ✅ {verdict}\nModel score (phishing): **{p:.2f}**")
    else:
        st.warning(f"### ⚠️ {verdict}\nModel score (phishing): **{p:.2f}**")
        for r in reasons:
            st.write("- " + r)
    st.caption(
        "The score is a raw model output and is not a calibrated probability. "
        "A 'legitimate' result is never a guarantee of safety - verify "
        "suspicious messages through an independent channel."
    )


def escalate_if_warnings(verdict, reasons, indicators):
    """Safety net for the dangerous error (phishing called legitimate).

    If the model says 'legitimate' but serious rule-based warning signs were
    found (credential requests, IP-address host, '@' trick, punycode), the
    result is changed to 'Uncertain - manual review'. It never forces 'phishing'.
    """
    if verdict == LEGIT:
        high = [msg for sev, msg in indicators if sev == "high"]
        if high:
            return UNCERTAIN, list(reasons) + [
                "The model scored it as legitimate, but serious warning signs were found: "
                + " ".join(high)
            ]
    return verdict, reasons


def show_indicators(items):
    if not items:
        st.write("No rule-based warning signals found.")
    for sev, msg in items:
        st.write(f"{SEVERITY_ICON[sev]} {msg}")


def sidebar_settings():
    st.sidebar.header("Settings")
    low, high = st.sidebar.slider(
        "Uncertain band (scores in this range → manual review)",
        0.0, 1.0, (0.35, 0.65), 0.05,
    )
    st.sidebar.caption(
        "Widening the band sends more messages to manual review - safer, "
        "but more work for a human."
    )
    return low, high


# ----------------------------------------------------------- URL tab ------
def url_tab(low, high):
    st.subheader("Check a URL")
    if url_models is None:
        st.info("URL model not found. Run `python train_url.py` first (see README).")
        return

    st.caption("The URL is analysed as text only. It is never opened or requested.")
    primary = st.selectbox("Model used for the verdict & explanation",
                           list(url_models), index=list(url_models).index("Random Forest")
                           if "Random Forest" in url_models else 0, key="url_primary")
    url = st.text_input("Paste a URL", placeholder="https://github.com/scikit-learn  or  http://paypa1-secure-login.example.top/verify", key="url_in")
    if st.button("Analyse URL"):
        st.session_state["url_done"] = url.strip()

    target = st.session_state.get("url_done")
    if not target:
        return

    feats = extract_url_feature_dict(target)
    row = pd.DataFrame([[feats[n] for n in FEATURE_NAMES]], columns=FEATURE_NAMES)
    probs = {n: float(m.predict_proba(row)[0, 1]) for n, m in url_models.items()}
    verdict, reasons = decide(probs[primary], probs, low, high)
    url_indicators = url_risk_indicators(target, feats)
    verdict, reasons = escalate_if_warnings(verdict, reasons, url_indicators)

    st.markdown(f"**Analysed:** `{target}`")
    show_verdict(verdict, probs[primary], reasons)

    st.markdown("#### Model comparison")
    st.dataframe(pd.DataFrame({
        "Model": list(probs),
        "Phishing score": [round(v, 3) for v in probs.values()],
        "Prediction": ["Phishing" if v >= 0.5 else "Legitimate" for v in probs.values()],
    }).set_index("Model"))

    st.markdown("#### Risk indicators (rules)")
    show_indicators(url_indicators)

    st.markdown(f"#### Which features moved the **{primary}** score?")
    exp = occlusion_explanation(url_models[primary], feats, url_meta["legit_baseline"], FEATURE_NAMES)
    if exp.empty:
        st.write("All features look like a typical legitimate URL, so none stands out.")
    else:
        top = exp.head(8).iloc[::-1]
        fig = px.bar(
            top, x="effect_on_phishing_score", y="feature", orientation="h",
            color=top["effect_on_phishing_score"] > 0,
            color_discrete_map={True: "#d62728", False: "#2ca02c"},
            labels={"effect_on_phishing_score": "Change in phishing score if this feature looked normal",
                    "feature": ""},
        )
        fig.update_layout(showlegend=False, height=380, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig)
        st.caption(
            "Red bars pushed the score towards phishing; green bars pushed it towards legitimate. "
            "Computed by replacing each feature with the typical value of legitimate training URLs "
            "(one feature at a time, so interactions are ignored)."
        )
        st.dataframe(exp.head(8).round(3))
    with st.expander("All extracted feature values"):
        st.dataframe(row.T.rename(columns={0: "value"}))


# --------------------------------------------------------- email tab ------
def email_tab(low, high):
    st.subheader("Check an email")
    if email_models is None:
        st.info("Email model not found. Run `python train_email.py --demo` (quick test) "
                "or `python train_email.py --csv your_data.csv` first (see README).")
        return
    if email_meta and email_meta.get("demo_mode"):
        st.warning("These email models were trained on a tiny DEMO dataset - results are not reliable. "
                   "Retrain with a real dataset.")

    primary = st.selectbox("Model used for the verdict & explanation", list(email_models),
                           key="email_primary")
    text = st.text_area("Paste the email text (subject + body)", height=200, key="email_in")
    if st.button("Analyse email"):
        st.session_state["email_done"] = text.strip()

    target = st.session_state.get("email_done")
    if not target:
        return

    probs = {n: float(m.predict_proba([target])[0, 1]) for n, m in email_models.items()}
    verdict, reasons = decide(probs[primary], probs, low, high)
    email_indicators = email_risk_indicators(target)
    verdict, reasons = escalate_if_warnings(verdict, reasons, email_indicators)
    show_verdict(verdict, probs[primary], reasons)

    st.markdown("#### Model comparison")
    st.dataframe(pd.DataFrame({
        "Model": list(probs),
        "Phishing score": [round(v, 3) for v in probs.values()],
        "Prediction": ["Phishing" if v >= 0.5 else "Legitimate" for v in probs.values()],
    }).set_index("Model"))

    st.markdown("#### Risk indicators (rules)")
    show_indicators(email_indicators)

    st.markdown(f"#### Words that influenced the **{primary}** score")
    contrib = email_term_contributions(email_models[primary], target)
    if contrib.empty:
        st.write("None of the words in this text were known to the model.")
    else:
        shown = pd.concat([contrib.head(6), contrib.tail(6)]).drop_duplicates("term")
        shown = shown.sort_values("contribution")
        fig = px.bar(
            shown, x="contribution", y="term", orientation="h",
            color=shown["contribution"] > 0,
            color_discrete_map={True: "#d62728", False: "#2ca02c"},
            labels={"contribution": "Contribution to phishing score", "term": ""},
        )
        fig.update_layout(showlegend=False, height=420, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig)
        st.caption("Red = word pushed towards phishing, green = towards legitimate. "
                   "URLs, numbers and e-mail addresses are replaced by the tokens "
                   "'urltoken', 'numtoken' and 'emailtoken' during cleaning.")


# ------------------------------------------------------ evaluation tab ----
def evaluation_section(title, metrics_file, fn_file, fn_col, importance=None):
    st.subheader(title)
    metrics = load_json(metrics_file)
    if metrics is None:
        st.info("No metrics yet - train this model first.")
        return

    rows = []
    for name, m in metrics.items():
        rows.append({
            "Model": name,
            "Accuracy": m["accuracy"], "Precision": m["precision"],
            "Recall": m["recall"], "F1": m["f1"],
            "False-negative rate": m["false_negative_rate"],
            "Brier score": m.get("brier_score"),
            "Latency (ms / item)": m.get("latency_ms_single"),
            "Test samples": m["test_samples"],
        })
    table = pd.DataFrame(rows).set_index("Model")
    st.dataframe(table.round(4))
    st.caption("Positive class = phishing. False-negative rate = share of phishing "
               "messages wrongly called legitimate (the most dangerous error). "
               "Brier score: lower is better (measures how well scores behave as probabilities).")

    long = table.reset_index().melt(id_vars="Model", value_vars=["Precision", "Recall", "F1"],
                                    var_name="Metric", value_name="Score")
    fig = px.bar(long, x="Model", y="Score", color="Metric", barmode="group", range_y=[0, 1.05])
    fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig)

    st.markdown("**Confusion matrices** (rows = true class, columns = predicted)")
    cols = st.columns(len(metrics))
    for col, (name, m) in zip(cols, metrics.items()):
        cm = np.array(m["confusion_matrix"])
        f = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                      x=["Pred. legit", "Pred. phishing"], y=["True legit", "True phishing"])
        f.update_layout(title=name, coloraxis_showscale=False, height=300,
                        margin=dict(l=0, r=0, t=40, b=0))
        col.plotly_chart(f)

    fn = load_csv(fn_file)
    st.markdown("**Phishing messages classified as legitimate (false negatives)**")
    if fn is None or fn.empty:
        st.write("None in the test set for the reference model.")
    else:
        st.caption("Study these to see what the model misses. Do not open any link shown here.")
        st.dataframe(fn.head(50))

    if importance:
        st.markdown("**Global feature importance (Random Forest)**")
        imp = pd.Series(importance).sort_values().tail(12)
        f = px.bar(imp, orientation="h", labels={"value": "Importance", "index": ""})
        f.update_layout(showlegend=False, height=380, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(f)


def evaluation_tab():
    evaluation_section("URL models", "url_metrics.json", "url_false_negatives.csv", "url",
                       importance=(url_meta or {}).get("rf_importance"))
    if url_meta:
        st.caption(f"URL dataset: {url_meta['n_phishing_total']:,} phishing and "
                   f"{url_meta['n_legit_total']:,} legitimate URLs after duplicate removal; "
                   f"{url_meta['n_test']:,} held out for testing.")
        if url_meta.get("augmented_legitimate"):
            st.info(
                "**Dataset bias fix.** In PhiUSIIL every legitimate URL is exactly "
                "`https://www.<domain>` with no path, so a model trained on it flags normal URLs "
                "such as `https://github.com/scikit-learn` as phishing (99%+ test accuracy, but "
                "useless on real URLs). The legitimate class was therefore replaced by synthetic "
                "variants of the same domains (http/https, www on/off, subdomains, paths, query "
                "strings) and the features ignore scheme and 'www'. Scores on this harder, "
                "augmented test split are lower but more honest."
            )
        sanity = load_csv("url_sanity.csv")
        if sanity is not None and not sanity.empty:
            st.markdown("**Real-world sanity check - 66 hand-picked URLs** "
                        "(46 typical legitimate URLs, 20 fabricated phishing-style URLs)")
            show = sanity.copy()
            show["true class"] = show["label"].map({0: "legitimate", 1: "phishing"})
            for name in url_models or {}:
                col = f"{name} score"
                if col in show:
                    show[f"{name} verdict"] = (show[col] >= 0.5).map({True: "phishing", False: "legitimate"})
            cols = ["url", "true class"] + [c for c in show.columns
                                            if c.endswith("score") or c.endswith("verdict")]
            st.dataframe(show[cols])
            acc = {name: float(((show[f"{name} score"] >= 0.5).astype(int) == show["label"]).mean())
                   for name in (url_models or {}) if f"{name} score" in show}
            st.caption("Accuracy on this small hand-picked set: "
                       + ", ".join(f"{k} {v:.0%}" for k, v in acc.items())
                       + ". This is a sanity check, not an independent benchmark: the set is tiny and "
                         "the augmentation was tuned after looking at it. Do not open any of these URLs.")
    st.divider()
    evaluation_section("Email models", "email_metrics.json", "email_false_negatives.csv", "text_preview")
    if email_meta:
        st.caption(f"Email dataset file: {email_meta['source_file']} "
                   f"({email_meta['n_emails']:,} emails, {email_meta['n_test']:,} in the test set)."
                   + (" DEMO DATA - not for reporting." if email_meta.get("demo_mode") else ""))


def about_tab():
    st.subheader("How it works")
    st.markdown(
        """
1. **Features** - URLs become 20 numeric signals (length, subdomains, symbols, IP host, HTTPS...).
   Emails become TF-IDF word features after cleaning.
2. **Models** - Logistic Regression (baseline) is compared with Random Forest (URLs) or
   Naive Bayes (emails). A majority-class baseline is also reported.
3. **Explanations** - rule-based risk indicators, plus model-based feature/word contributions.
4. **Manual review** - scores near 0.5, disagreement between models, or a 'legitimate' score combined
   with serious rule-based warning signs (credential requests, IP-address host, '@' trick, punycode)
   give an *Uncertain* result.

### Limitations and responsible use
- Models can fail on new phishing tactics or when real-world data differs from the training data.
- A model score is not a guarantee of safety, and HTTPS alone does not make a site trustworthy.
- This tool is educational. Always verify suspicious messages through an independent, trusted channel.
- Never open suspicious URLs while testing - this app does not.
"""
    )


# ---------------------------------------------------------------- main ----
st.title("🛡️ Explainable AI Phishing Detector")
st.caption("Educational tool - predictions can be wrong. Never rely on it as your only defence.")
low_band, high_band = sidebar_settings()
tab_url, tab_email, tab_eval, tab_about = st.tabs(
    ["🔗 URL check", "✉️ Email check", "📊 Evaluation dashboard", "ℹ️ About & limitations"]
)
with tab_url:
    url_tab(low_band, high_band)
with tab_email:
    email_tab(low_band, high_band)
with tab_eval:
    evaluation_tab()
with tab_about:
    about_tab()