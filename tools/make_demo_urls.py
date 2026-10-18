"""Generate a SYNTHETIC URL dataset for smoke-testing the pipeline offline.

    python tools/make_demo_urls.py
    python train_url.py --csv data/demo_urls.csv

The data is randomly generated, so the metrics are NOT meaningful. Use the
real PhiUSIIL dataset (python train_url.py) for your project results.
"""
import csv
import random
from pathlib import Path

random.seed(7)
LEGIT_HOSTS = ["www.wikipedia.org", "www.python.org", "github.com", "www.bbc.co.uk",
               "stackoverflow.com", "www.nytimes.com", "docs.djangoproject.com",
               "www.amazon.in", "en.wikipedia.org", "scikit-learn.org", "pypi.org",
               "www.coursera.org", "www.khanacademy.org", "news.ycombinator.com"]
LEGIT_PATHS = ["", "/", "/about", "/docs/stable/index.html", "/wiki/Machine_learning",
               "/questions/12345/how-to-sort", "/news/technology", "/courses", "/blog/2024/post"]
WORDS = ["secure", "login", "verify", "account", "update", "bank", "paypal", "signin", "wallet"]
TLDS = ["com", "net", "info", "xyz", "top", "biz"]


def legit():
    scheme = random.choices(["https", "http"], [0.9, 0.1])[0]
    return f"{scheme}://{random.choice(LEGIT_HOSTS)}{random.choice(LEGIT_PATHS)}"


def phish():
    kind = random.randint(0, 5)
    w1, w2 = random.sample(WORDS, 2)
    tld = random.choice(TLDS)
    if kind == 0:
        ip = ".".join(str(random.randint(1, 254)) for _ in range(4))
        return f"http://{ip}/{w1}/{w2}.php?id={random.randint(100000, 999999)}"
    if kind == 1:
        return f"http://{w1}-{w2}-{random.randint(10, 99)}.example-{w1}.{tld}/{w2}?session={random.randint(10**6, 10**9)}"
    if kind == 2:
        return f"http://www.google.com@{w1}-{w2}.{tld}/index.html"
    if kind == 3:
        return f"https://{w1}.{w2}.account.verify.{random.choice(WORDS)}-{random.randint(1, 99)}.{tld}/login"
    if kind == 4:
        return f"http://bit.ly/{''.join(random.choices('abcdefghijk0123456789', k=7))}"
    return f"https://xn--pypal-4ve.{tld}/{w1}/{w2}//confirm%20account?token={random.randint(10**7, 10**9)}"


rows = [(legit() + (f"?ref={random.randint(1, 9999)}" if random.random() < 0.3 else ""), 0) for _ in range(1500)]
rows += [(phish(), 1) for _ in range(1500)]
random.shuffle(rows)
out = Path(__file__).resolve().parent.parent / "data" / "demo_urls.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["url", "label"])
    w.writerows(rows)
print(f"Wrote {len(rows)} synthetic rows to {out}")
