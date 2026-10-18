"""URL feature extraction.

The SAME function is used for training (train_url.py) and for the live
scanner (app.py), so there is no train/serve mismatch. Nothing here ever
opens, resolves or downloads a URL - it only inspects the text.

Design note (important for the project report)
----------------------------------------------
In the PhiUSIIL dataset EVERY legitimate URL has the exact form
"https://www.<domain>" with no path. A model trained on raw counts
therefore learns "www + https + no path = legitimate" and wrongly flags real
sites such as https://github.com/scikit-learn. To stop the model from using
these dataset artefacts we

  1. compute length / dot / hyphen / digit features on a NORMALISED URL
     (scheme and leading "www." removed, lower-cased, trailing "/" removed), and
  2. augment the legitimate class in train_url.py with realistic variants
     (http/https, with/without www, subdomains, paths, query strings).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from urllib.parse import urlparse

FEATURE_NAMES = [
    "url_length",
    "hostname_length",
    "path_length",
    "count_dots",
    "count_hyphens",
    "count_at",
    "count_question",
    "count_equal",
    "count_ampersand",
    "count_percent",
    "count_digits",
    "count_subdomains",
    "has_ip_address",
    "has_https",
    "has_punycode",
    "has_shortener_hint",
    "has_suspicious_keyword",
    "has_double_slash_in_path",
    "has_encoded_chars",
    "has_long_numeric_run",
    "count_path_segments",
    "digit_ratio",
    "longest_alnum_run",
    "char_entropy",
]

FEATURE_DESCRIPTIONS = {
    "url_length": "Characters in the URL (scheme and 'www.' ignored)",
    "hostname_length": "Length of the domain/host part",
    "path_length": "Length of the path after the domain",
    "count_dots": "Number of '.' characters",
    "count_hyphens": "Number of '-' characters",
    "count_at": "Number of '@' characters (can hide the real destination)",
    "count_question": "Number of '?' characters (query strings)",
    "count_equal": "Number of '=' characters",
    "count_ampersand": "Number of '&' characters",
    "count_percent": "Number of '%' characters (encoding)",
    "count_digits": "Number of digits in the URL",
    "count_subdomains": "Number of subdomain levels before the main domain ('www' ignored)",
    "has_ip_address": "Host is an IP address instead of a domain name",
    "has_https": "URL starts with https:// (NOT proof of safety)",
    "has_punycode": "Host contains 'xn--' (possible lookalike characters)",
    "has_shortener_hint": "Host is a known URL-shortening service",
    "has_suspicious_keyword": "Contains words common in phishing (login, verify...)",
    "has_double_slash_in_path": "Contains '//' inside the path (redirect trick)",
    "has_encoded_chars": "Contains %XX encoded characters",
    "has_long_numeric_run": "Contains a run of 6+ digits in a row",
    "count_path_segments": "Number of '/'-separated parts in the path",
    "digit_ratio": "Share of characters that are digits",
    "longest_alnum_run": "Longest unbroken run of letters/digits (random tokens are long)",
    "char_entropy": "Randomness of the characters (high = random-looking text)",
}

SUSPICIOUS_KEYWORDS = (
    "login", "signin", "verify", "secure", "account", "update", "bank",
    "confirm", "password", "paypal", "wallet", "billing", "suspend",
    "unlock", "webscr", "ebayisapi",
)

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "shorturl.at", "tiny.cc", "rb.gy", "lnkd.in",
    "t.ly", "bl.ink",
}

_MULTI_PART_TLDS = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "co.in", "net.in",
    "org.in", "ac.in", "gov.in", "co.jp", "com.br", "com.cn", "co.za",
}

_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
_WWW = re.compile(r"^www\d{0,3}\.", re.I)
_IPV4 = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
_HEX_IP = re.compile(r"^0x[0-9a-f]+$", re.I)
_DECIMAL_IP = re.compile(r"^\d{8,10}$")
_ENCODED = re.compile(r"%[0-9a-fA-F]{2}")
_LONG_NUM = re.compile(r"\d{6,}")
_ALNUM_RUN = re.compile(r"[a-z0-9]+")


def _prepare(url: str) -> str:
    """Strip whitespace and add http:// if the user pasted a bare domain."""
    raw = str(url).strip()
    return raw if _SCHEME.match(raw) else "http://" + raw


def _normalise(full: str) -> str:
    """Remove scheme + leading www, lower-case, drop one trailing slash."""
    norm = _SCHEME.sub("", full)
    norm = _WWW.sub("", norm)
    norm = norm.lower()
    return norm[:-1] if norm.endswith("/") else norm


def _host_and_path(full: str) -> tuple[str, str]:
    try:
        parsed = urlparse(full)
        return (parsed.hostname or "").lower(), parsed.path or ""
    except ValueError:  # e.g. malformed IPv6 brackets
        rest = _SCHEME.sub("", full)
        host = re.split(r"[/?#]", rest, maxsplit=1)[0].split("@")[-1]
        return host.lower(), ""


def _is_ip(host: str) -> bool:
    return bool(
        _IPV4.match(host)
        or _HEX_IP.match(host)
        or _DECIMAL_IP.match(host)
        or ":" in host  # IPv6
    )


def _count_subdomains(bare_host: str) -> int:
    if not bare_host or _is_ip(bare_host):
        return 0
    labels = [p for p in bare_host.split(".") if p]
    base = 3 if ".".join(labels[-2:]) in _MULTI_PART_TLDS else 2
    return max(0, len(labels) - base)


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in Counter(text).values())


def matched_keywords(url: str) -> list[str]:
    low = _normalise(_prepare(url))
    return [k for k in SUSPICIOUS_KEYWORDS if k in low]


def extract_url_features(url: str) -> list[float]:
    """Return the feature values in the order of FEATURE_NAMES."""
    full = _prepare(url)
    norm = _normalise(full)
    host, path = _host_and_path(full)
    bare_host = _WWW.sub("", host)
    path = path.rstrip("/")

    is_shortener = any(
        bare_host == s or bare_host.endswith("." + s) for s in SHORTENERS
    )
    n = max(len(norm), 1)
    digits = sum(ch.isdigit() for ch in norm)
    runs = _ALNUM_RUN.findall(norm)

    values = {
        "url_length": len(norm),
        "hostname_length": len(bare_host),
        "path_length": len(path),
        "count_dots": norm.count("."),
        "count_hyphens": norm.count("-"),
        "count_at": norm.count("@"),
        "count_question": norm.count("?"),
        "count_equal": norm.count("="),
        "count_ampersand": norm.count("&"),
        "count_percent": norm.count("%"),
        "count_digits": digits,
        "count_subdomains": _count_subdomains(bare_host),
        "has_ip_address": int(_is_ip(host)),
        "has_https": int(full.lower().startswith("https://")),
        "has_punycode": int("xn--" in host),
        "has_shortener_hint": int(is_shortener),
        "has_suspicious_keyword": int(bool(matched_keywords(full))),
        "has_double_slash_in_path": int("//" in path),
        "has_encoded_chars": int(bool(_ENCODED.search(norm))),
        "has_long_numeric_run": int(bool(_LONG_NUM.search(norm))),
        "count_path_segments": len([s for s in path.split("/") if s]),
        "digit_ratio": digits / n,
        "longest_alnum_run": max((len(r) for r in runs), default=0),
        "char_entropy": _entropy(norm),
    }
    return [values[name] for name in FEATURE_NAMES]


def extract_url_feature_dict(url: str) -> dict:
    return dict(zip(FEATURE_NAMES, extract_url_features(url)))
