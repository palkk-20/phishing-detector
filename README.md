# Explainable AI Phishing Detector

A Streamlit web app that classifies a **URL** or an **email text** as likely phishing, likely
legitimate, or **uncertain (manual review)** and explains which signals influenced the result.

> Educational project. A model score is not a guarantee of safety. The app only analyses text -
> it never opens or visits a URL.

## Features
- URL analysis (24 features) and email-text classification (TF-IDF)
- Model comparison: Logistic Regression (baseline) vs Random Forest (URLs) / Naive Bayes (emails), plus a majority-class baseline
- Explanations: rule-based risk indicators + feature / word contributions
- Evaluation dashboard: precision, recall, F1, confusion matrix, false-negative rate, Brier score, latency, missed-phishing list, real-world sanity check
- Uncertain / manual-review outcome (score near 0.5 or models disagree; adjustable in the sidebar)

## Project structure
```
app.py              Streamlit app
train_url.py        trains the URL models   (options: --csv, --refresh, --no-augment)
train_email.py      trains the email models (options: --csv, --demo, --text-col, --label-col, --flip-labels)
requirements.txt
src/                features.py, augment.py, explain.py, decision.py, evaluation.py, text_utils.py
data/               phiusiil_urls.csv (cached URL dataset), sample_emails.csv (tiny demo set)
artifacts/          trained models + metrics
tests/              test_features.py, sanity_urls.csv (66 hand-picked URLs)
tools/              make_demo_urls.py (synthetic data for offline smoke tests)
```

## Setup and run
```bash
python -m venv venv
source venv/Scripts/activate        # Git Bash on Windows  (cmd: venv\Scripts\activate)
pip install -r requirements.txt

streamlit run app.py                # trained models are already included in artifacts/
```
To retrain (optional):
```bash
python train_url.py                                  # uses data/phiusiil_urls.csv (no download needed)
curl -L -o data/emails.csv "https://zenodo.org/records/8339691/files/CEAS_08.csv?download=1"
python train_email.py --csv data/emails.csv --text-col body --label-col label
```

## Datasets and licences
- **URLs:** PhiUSIIL Phishing URL Dataset, UCI Machine Learning Repository (id 967). Check its licence and citation on the UCI page.
- **Emails:** "Phishing Email Curated Datasets" (CEAS_08.csv), Champa, Rabbi and Zibran, Zenodo, DOI 10.5281/zenodo.8339691, CC BY 4.0. Cite the record and the two papers listed on its page. It mixes spam and phishing; labels follow the dataset.
- `data/sample_emails.csv` is 40 hand-written emails for pipeline tests only.

## Important finding: dataset bias in PhiUSIIL (and the fix)
In PhiUSIIL **every one of the 134,850 legitimate URLs is exactly `https://www.<domain>` with no path**,
while phishing URLs are 49 % https, 41 % www and 60 % have a path. A model trained on raw features
reaches 99 %+ test accuracy by learning "www + https + no path = legitimate", but then flags real
URLs such as `https://github.com` or `https://github.com/scikit-learn` as phishing.

Fix (see `src/features.py`, `src/augment.py`, `train_url.py`):
1. Features are computed on a normalised URL: scheme and leading `www.` removed, lower-cased, trailing `/` removed.
   Four more features were added (path segments, digit ratio, longest alphanumeric run, character entropy).
2. The legitimate class is **augmented**: each legitimate URL is replaced by one synthetic variant of the same
   domain (http/https, www on/off, optional subdomain, path, query string, fragment). Phishing URLs are unchanged.
3. `python train_url.py --no-augment` reproduces the biased behaviour so you can show before/after.

Results (80/20 stratified split, class 1 = phishing):

| Model | Accuracy | Precision | Recall | F1 | Phishing missed |
|---|---|---|---|---|---|
| Majority-class baseline | 0.573 | 0 | 0 | 0 | 100 % |
| Logistic Regression | 0.797 | 0.741 | 0.807 | 0.773 | 19.3 % |
| Random Forest | 0.874 | 0.827 | 0.891 | 0.858 | 10.9 % |

These scores are lower than the 99 % obtained on the raw dataset because the task is now harder and more honest.
After the fix `https://www.github.com`, `https://github.com` and `https://www.github.com/scikit-learn` are all classified legitimate.

Honest caveats: the augmented legitimate URLs are **synthetic**; the 66-URL sanity check (`tests/sanity_urls.csv`)
is small and the augmentation was tuned after looking at it, so it is not an independent benchmark; URL
lexical features alone cannot separate every case (short legitimate URLs such as `https://x.com/home` or
`https://aws.amazon.com/s3/` can still be flagged). This is exactly why the app has an *Uncertain - manual review* outcome.

## Limitations
- May fail on new phishing tactics or on data from a different distribution than the training set.
- Scores are not calibrated probabilities. HTTPS alone does not make a site trustworthy.
- URL explanations change one feature at a time and ignore interactions.
- The email data is from 2008; modern or casual emails can land in the uncertain zone.
