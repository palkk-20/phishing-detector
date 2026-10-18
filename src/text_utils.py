"""Text cleaning for the email model (used inside the TF-IDF vectoriser)."""
import re

_HTML = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_DIGITS = re.compile(r"\d+")
_NON_LETTER = re.compile(r"[^a-z\s]")
_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lower-case, strip HTML, and replace URLs / emails / numbers by tokens.

    Replacing a URL with the token 'urltoken' lets the model learn that
    'a link is present' without memorising individual domains.
    """
    t = str(text)
    t = _HTML.sub(" ", t)
    t = _URL.sub(" urltoken ", t)
    t = _EMAIL.sub(" emailtoken ", t)
    t = t.lower()
    t = _DIGITS.sub(" numtoken ", t)
    t = _NON_LETTER.sub(" ", t)
    return _SPACES.sub(" ", t).strip()
