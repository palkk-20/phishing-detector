"""Explanations: rule-based risk indicators + model-based feature effects."""
from __future__ import annotations

import re

import pandas as pd

from .features import FEATURE_DESCRIPTIONS, matched_keywords


# ---------------------------------------------------------------- URL ----
def url_risk_indicators(url: str, f: dict) -> list[tuple[str, str]]:
    """Human-readable warning signals as (severity, message)."""
    out = []
    if f["has_ip_address"]:
        out.append(("high", "The host is an IP address, not a normal domain name."))
    if f["count_at"]:
        out.append(("high", "Contains '@' - everything before it can hide the real destination."))
    if f["has_punycode"]:
        out.append(("high", "Punycode host (xn--) - may imitate a known site with lookalike characters."))
    if f["has_shortener_hint"]:
        out.append(("medium", "A URL shortener hides the final destination."))
    kws = matched_keywords(url)
    if kws:
        out.append(("medium", "Contains words often used in phishing: " + ", ".join(kws) + "."))
    if f["count_subdomains"] >= 3:
        out.append(("medium", f"Many subdomain levels ({f['count_subdomains']})."))
    if f["count_hyphens"] >= 3:
        out.append(("medium", f"Many hyphens ({f['count_hyphens']}) - typical of made-up lookalike domains."))
    if f["url_length"] > 75:
        out.append(("low", f"Long URL ({f['url_length']} characters)."))
    if f["has_double_slash_in_path"]:
        out.append(("low", "Contains '//' in the path (possible redirect trick)."))
    if f["has_encoded_chars"]:
        out.append(("low", "Contains %-encoded characters that can obscure the real text."))
    if f["has_long_numeric_run"]:
        out.append(("low", "Contains a long run of digits."))
    if f["has_https"]:
        out.append(("info", "Uses HTTPS. Many phishing sites use HTTPS too, so this does NOT prove the site is safe."))
    else:
        out.append(("low", "Does not use HTTPS."))
    return out


def occlusion_explanation(model, features: dict, baseline: dict, feature_names) -> pd.DataFrame:
    """Model-agnostic local explanation.

    For each feature we replace its value with the typical value seen in
    legitimate training URLs and measure how much the phishing probability
    drops. effect > 0  => this feature pushed the score towards phishing.
    (Single-feature changes ignore interactions - it is an approximation.)
    """
    base_row = [features[n] for n in feature_names]
    rows = [base_row]
    for i, n in enumerate(feature_names):
        r = list(base_row)
        r[i] = baseline[n]
        rows.append(r)
    X = pd.DataFrame(rows, columns=list(feature_names))
    p = model.predict_proba(X)[:, 1]
    df = pd.DataFrame({
        "feature": list(feature_names),
        "meaning": [FEATURE_DESCRIPTIONS[n] for n in feature_names],
        "value": base_row,
        "typical_legit_value": [baseline[n] for n in feature_names],
        "effect_on_phishing_score": p[0] - p[1:],
    })
    df = df[df["value"] != df["typical_legit_value"]]
    return df.reindex(
        df["effect_on_phishing_score"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)


# -------------------------------------------------------------- email ----
_URGENCY = ("urgent", "immediately", "within 24 hours", "act now", "expires",
            "suspended", "final notice", "last warning", "limited time")
_CREDENTIALS = ("password", "verify your", "confirm your", "login", "log in",
                "sign in", "account number", "ssn", "pin ", "credit card", "otp")
_MONEY = ("prize", "winner", "lottery", "refund", "claim your", "gift card",
          "wire transfer", "invoice", "payment")
_GENERIC = ("dear customer", "dear user", "dear member", "dear client",
            "valued customer", "dear account holder")


def _found(text_low: str, words) -> list[str]:
    """Whole-word / whole-phrase match (so 'pin' does not match 'shipping')."""
    hits = []
    for w in words:
        w = w.strip()
        if re.search(r"(?<![a-z0-9])" + re.escape(w) + r"(?![a-z0-9])", text_low):
            hits.append(w)
    return hits


def email_risk_indicators(text: str) -> list[tuple[str, str]]:
    low = text.lower()
    out = []
    urls = re.findall(r"https?://\S+|www\.\S+", text, flags=re.I)
    if urls:
        out.append(("medium", f"Contains {len(urls)} link(s). Check where each link really goes before clicking."))
    if any(re.match(r"https?://\d{1,3}(\.\d{1,3}){3}", u, re.I) for u in urls):
        out.append(("high", "A link points directly to an IP address."))
    for label, words, sev in (
        ("Urgency / pressure wording", _URGENCY, "medium"),
        ("Asks for credentials or personal data", _CREDENTIALS, "high"),
        ("Money / prize / payment wording", _MONEY, "medium"),
        ("Generic greeting instead of your name", _GENERIC, "low"),
    ):
        hit = _found(low, words)
        if hit:
            out.append((sev, f"{label}: " + ", ".join(hit[:5]) + "."))
    words = re.findall(r"[A-Za-z]{3,}", text)
    if words:
        caps = sum(w.isupper() for w in words) / len(words)
        if caps > 0.15:
            out.append(("low", f"Lots of ALL-CAPS words ({caps:.0%})."))
    if text.count("!") >= 3:
        out.append(("low", "Many exclamation marks."))
    return out


def email_term_contributions(pipeline, text: str) -> pd.DataFrame:
    """Per-word contribution to the phishing score (positive = phishing).

    Works for Logistic Regression (coef_) and Naive Bayes
    (difference of class log-probabilities).
    """
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    vec = tfidf.transform([text]).tocoo()
    if hasattr(clf, "coef_"):
        w = clf.coef_[0]
    else:
        w = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    names = tfidf.get_feature_names_out()
    rows = [(names[j], float(v * w[j])) for j, v in zip(vec.col, vec.data)]
    df = pd.DataFrame(rows, columns=["term", "contribution"])
    return df.sort_values("contribution", ascending=False).reset_index(drop=True)