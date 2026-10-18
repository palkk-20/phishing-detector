"""Augmentation of the LEGITIMATE URL class.

Why: in PhiUSIIL every legitimate URL is literally "https://www.<domain>"
(no path, always https, always www). That is a sampling artefact, not a
property of real legitimate sites, and a model trained on it flags normal
URLs such as https://github.com/scikit-learn as phishing.

What: every legitimate URL is replaced by ONE realistic variant of the same
domain (scheme http/https, www on/off, optional subdomain, optional path,
optional query string). Phishing URLs are never touched. The variants are
synthetic - say so in the report.
"""
from __future__ import annotations

import random
import re

_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
_WWW = re.compile(r"^www\.", re.I)

WORDS = (
    "about contact products product services service blog news docs guide help support "
    "store shop pricing features download downloads courses course learn tutorial "
    "events team careers jobs press media gallery faq terms privacy resources "
    "research publications projects portfolio stories article articles category "
    "reviews offers search results profile settings community forum topics "
    "getting-started introduction overview documentation release-notes whats-new "
    "python data science machine learning health travel sports music design"
).split()
LOCALES = ("en", "us", "in", "uk", "en-us", "en-in", "en-gb", "fr", "de", "es", "hi", "v1", "v2", "3", "latest", "stable")
ACTIONS = ("login", "signin", "account", "accounts", "support", "update", "billing", "password-reset", "verify-email", "secure")
DOTTED = ("urllib.parse", "os.path", "index", "intro", "overview", "v1.2.3", "3.12", "python_intro", "data_structures", "getting_started")
ALNUM = "abcdefghijklmnopqrstuvwxyz0123456789"
SUBDOMAINS = ("docs", "blog", "en", "support", "shop", "help", "news", "app", "store", "learn")
EXTENSIONS = (".html", ".htm", ".php", ".aspx", ".jsp")
QUERY_KEYS = ("id", "page", "q", "ref", "lang", "tab", "sort", "utm_source", "category")


def _slug(rng: random.Random) -> str:
    return "-".join(rng.choices(WORDS, k=rng.randint(1, 4)))


def _token(rng: random.Random) -> str:
    """Short random-looking id such as a video / product / post id (8-12 chars)."""
    return "".join(rng.choices(ALNUM, k=rng.randint(8, 12)))


def _segment(rng: random.Random) -> str:
    kind = rng.random()
    if kind < 0.34:
        return rng.choice(WORDS)
    if kind < 0.58:
        return _slug(rng)
    if kind < 0.70:
        return str(rng.randint(1, 99999999))
    if kind < 0.77:
        return f"{rng.choice(WORDS)}{rng.randint(1, 9999)}"
    if kind < 0.84:
        return rng.choice(DOTTED)
    if kind < 0.90:
        return rng.choice(ACTIONS)
    if kind < 0.95:
        return _token(rng)
    return "_".join(rng.choices(WORDS, k=rng.randint(2, 3)))


def _path(rng: random.Random) -> str:
    parts = [_segment(rng) for _ in range(rng.choices([1, 2, 3, 4], [40, 35, 18, 7])[0])]
    if rng.random() < 0.22:
        parts.insert(0, rng.choice(LOCALES))
    path = "/" + "/".join(parts)
    r = rng.random()
    if r < 0.18:
        path += rng.choice(EXTENSIONS)
    elif r < 0.30:
        path += "/"
    return path


def _query(rng: random.Random) -> str:
    n = rng.choices([1, 2, 3], [60, 30, 10])[0]
    pairs = []
    if rng.random() < 0.2:
        pairs.append(f"v={_token(rng)[:11]}")
    for key in rng.sample(QUERY_KEYS, n):
        val = str(rng.randint(1, 99999)) if rng.random() < 0.5 else rng.choice(WORDS)
        pairs.append(f"{key}={val}")
    return "?" + "&".join(pairs)


def augment_one(url: str, rng: random.Random) -> str:
    host = _SCHEME.sub("", url.strip())
    host = host.split("/")[0].split("?")[0]
    host = _WWW.sub("", host).lower()

    scheme = "https" if rng.random() < 0.72 else "http"
    if rng.random() < 0.12:
        prefix = rng.choice(SUBDOMAINS) + "."
    elif rng.random() < 0.5:
        prefix = "www."
    else:
        prefix = ""

    tail = ""
    if rng.random() < 0.55:
        tail = _path(rng)
        if rng.random() < 0.18:
            tail += _query(rng)
        elif rng.random() < 0.06:
            tail += "#" + rng.choice(WORDS)
    elif rng.random() < 0.3:
        tail = "/"
    return f"{scheme}://{prefix}{host}{tail}"


def augment_legitimate(urls, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    return [augment_one(u, rng) for u in urls]
