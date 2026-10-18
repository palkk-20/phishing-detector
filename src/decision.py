"""Turn a model score into a verdict, including the 'manual review' outcome."""

PHISHING = "Likely phishing"
LEGIT = "Likely legitimate"
UNCERTAIN = "Uncertain - manual review needed"


def decide(p_primary: float, all_probs: dict, low: float, high: float):
    """Return (verdict, reasons).

    Uncertain if the primary score is inside [low, high] OR the models
    disagree about the class. Otherwise follow the primary model.
    """
    reasons = []
    labels = {name: int(p >= 0.5) for name, p in all_probs.items()}
    if low <= p_primary <= high:
        reasons.append(
            f"The score ({p_primary:.2f}) is inside the uncertain band "
            f"[{low:.2f}, {high:.2f}]."
        )
    if len(set(labels.values())) > 1:
        reasons.append("The models disagree with each other.")
    if reasons:
        return UNCERTAIN, reasons
    return (PHISHING if p_primary > high else LEGIT), reasons
