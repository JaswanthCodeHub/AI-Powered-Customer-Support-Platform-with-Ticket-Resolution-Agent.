"""Evaluate category and severity on a separate labeled dataset."""

import csv
from pathlib import Path
from statistics import median
from time import perf_counter

from sklearn.metrics import classification_report

from classifier import classify, clean_text, load_classifier, load_dataset, predict_severity

EVALUATION_DATA = Path(__file__).resolve().parent / "data" / "evaluation.csv"


def evaluate(model=None):
    if model is None:
        model = load_classifier()

    with EVALUATION_DATA.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))

    training_texts, _ = load_dataset()
    training_keys = set(map(clean_text, training_texts))
    evaluation_keys = [clean_text(row["ticket"]) for row in rows]
    if len(set(evaluation_keys)) != len(rows) or training_keys.intersection(evaluation_keys):
        raise ValueError("Evaluation tickets must be unique and separate from training.")

    category_expected, category_predicted = [], []
    severity_expected, severity_predicted = [], []
    timings = []
    for row in rows:
        text = row["ticket"]
        started = perf_counter()
        category, _ = classify(model, text)
        severity = predict_severity(text)
        timings.append((perf_counter() - started) * 1000)
        category_expected.append(row["category"])
        category_predicted.append(category)
        if row.get("severity"):
            severity_expected.append(row["severity"])
        severity_predicted.append(severity)

    print(f"Kaggle held-out evaluation ({len(rows)} separate labeled tickets)")
    for name, expected, predicted, target in (
        ("Category", category_expected, category_predicted, 0.90),
        ("Severity", severity_expected, severity_predicted, 0.85),
    ):
        if not expected:
            print(f"\n{name}: unavailable; the Kaggle dataset has no severity labels.")
            continue
        correct = sum(a == b for a, b in zip(expected, predicted))
        accuracy = correct / len(expected)
        status = "PASS" if accuracy >= target else "BELOW TARGET"
        print(f"\n{name}: {correct}/{len(expected)} ({accuracy:.1%}), target {target:.0%}: {status}")
        print(classification_report(expected, predicted, zero_division=0))
    print(f"Median category + severity processing time: {median(timings):.2f} ms")
    print("Timing excludes server startup, HTTP, and SQLite.")
