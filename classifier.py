"""Train and evaluate the ticket category classifier."""

import re
import unicodedata
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

DATASET = Path(__file__).resolve().parent / "data" / "tickets.csv"
MODEL_PATH = DATASET.parents[1] / "models" / "ticket_classifier.pkl"
CATEGORIES = {"Hardware", "Access", "HR Support", "Miscellaneous", "Storage",
              "Purchase", "Internal Project", "Administrative rights"}
FILLER_WORDS = {"a", "an", "the", "i", "my", "is", "am", "me", "asap"}
CONTACT_OR_URL = re.compile(r"https?://\S+|[\w.+-]+@[\w.-]+\.\w+", re.I)


def normalize_text(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = CONTACT_OR_URL.sub(" ", text)
    text = re.sub(r"n['’]t\b", " not", text)
    return " ".join(re.findall(r"[^\W_]+", text))


def clean_text(text):
    return " ".join(token for token in normalize_text(text).split() if token not in FILLER_WORDS)


def load_dataset():
    data = pd.read_csv(DATASET, dtype=str, keep_default_na=False)
    texts = data["ticket"].tolist()
    labels = data["category"].tolist()
    cleaned = [clean_text(ticket) for ticket in texts]
    if set(labels) != CATEGORIES or "" in cleaned or len(set(cleaned)) != len(cleaned):
        raise ValueError("Expected unique, nonempty examples covering all ticket categories.")
    return texts, labels


def make_classifier():
    return make_pipeline(
        TfidfVectorizer(preprocessor=clean_text),
        LogisticRegression(),
    )


def train_classifier():
    texts, labels = load_dataset()
    model = make_classifier().fit(texts, labels)
    test = pd.read_csv(DATASET.with_name("evaluation.csv"), keep_default_na=False)
    return model, model.score(test.ticket, test.category)


def save_classifier(model):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=MODEL_PATH.parent, delete=False) as file:
        temporary = Path(file.name)
    try:
        joblib.dump(model, temporary)
        temporary.replace(MODEL_PATH)
    finally:
        temporary.unlink(missing_ok=True)
    return MODEL_PATH


def load_classifier():
    # Only load the artifact shipped with the project or produced by train_model.py.
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return train_classifier()[0]


def classify(model, text):
    probabilities = model.predict_proba([text])[0]
    best = int(probabilities.argmax())
    return str(model.classes_[best]), float(probabilities[best])


def predict_severity(ticket):
    text = ticket.lower()
    for severity, words in (
        ("Critical", ("server down", "entire company", "production down", "security breach")),
        ("High", ("urgent", "cannot work", "business stopped", "client meeting", "vpn not working")),
        ("Medium", ("slow", "error", "problem", "issue")),
    ):
        if any(word in text for word in words):
            return severity
    return "Low"


def calculate_priority(severity, business_impact):
    if severity in ("Critical", "High") and business_impact == "High":
        return "P1"
    return {"High": "P2", "Medium": "P3"}.get(severity, "P4")


def triage(text, business_impact=None):
    severity = predict_severity(text)
    if business_impact is None:
        return severity, {"Critical": "P1", "High": "P2", "Medium": "P3", "Low": "P4"}[severity]
    return severity, calculate_priority(severity, business_impact)


