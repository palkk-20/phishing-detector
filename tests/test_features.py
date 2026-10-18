"""Quick sanity tests. Run:  python tests/test_features.py   (or pytest)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import FEATURE_NAMES, extract_url_feature_dict, extract_url_features  # noqa: E402


def test_length_matches_names():
    assert len(extract_url_features("https://www.python.org")) == len(FEATURE_NAMES) == 24


def test_ip_and_at():
    f = extract_url_feature_dict("http://192.168.1.5/login@x")
    assert f["has_ip_address"] == 1 and f["count_at"] == 1 and f["has_suspicious_keyword"] == 1


def test_https_flag():
    assert extract_url_feature_dict("https://example.com")["has_https"] == 1
    assert extract_url_feature_dict("http://example.com")["has_https"] == 0
    assert extract_url_feature_dict("example.com")["has_https"] == 0


def test_subdomains():
    assert extract_url_feature_dict("https://a.b.c.example.com/x")["count_subdomains"] == 3
    assert extract_url_feature_dict("https://www.bbc.co.uk/")["count_subdomains"] == 0
    assert extract_url_feature_dict("https://news.bbc.co.uk/")["count_subdomains"] == 1


def test_punycode_shortener():
    assert extract_url_feature_dict("http://xn--pypal-4ve.com")["has_punycode"] == 1
    assert extract_url_feature_dict("https://bit.ly/abc123")["has_shortener_hint"] == 1


def test_www_and_trailing_slash_do_not_matter():
    a = extract_url_features("https://www.github.com")
    b = extract_url_features("https://github.com/")
    assert a == b


def test_augmentation_changes_legit_urls_only_in_form():
    from src.augment import augment_legitimate
    out = augment_legitimate(["https://www.example.org"] * 50, seed=1)
    assert any(u.startswith("http://") for u in out)
    assert any("/" in u.split("://", 1)[1] for u in out)
    assert all("example.org" in u for u in out)


def test_malformed_does_not_crash():
    extract_url_features("http://[bad")
    extract_url_features("")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
