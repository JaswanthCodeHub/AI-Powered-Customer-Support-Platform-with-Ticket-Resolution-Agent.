"""Prepare a reproducible Kaggle train/test split without duplicate text leakage."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from classifier import clean_text

ROOT = Path(__file__).resolve().parent


def prepare():
    raw = pd.read_csv(ROOT / "data/raw/all_tickets_processed_improved_v3.csv",
                      dtype=str, keep_default_na=False)
    data = raw.rename(columns={"Document": "ticket", "Topic_group": "category"})
    data = data[["ticket", "category"]].copy()
    data["key"] = data.ticket.map(clean_text)
    data = data[data.key.ne("") & data.category.ne("")]
    ambiguous = data.groupby("key").category.nunique()
    data = data[~data.key.isin(ambiguous[ambiguous > 1].index)]
    data = data.drop_duplicates("key").drop(columns="key")
    train, test = train_test_split(data, test_size=0.2, random_state=42,
                                   stratify=data.category)
    train.to_csv(ROOT / "data/tickets.csv", index=False)
    test.to_csv(ROOT / "data/evaluation.csv", index=False)
    print(f"Raw rows: {len(raw)}; cleaned unique rows: {len(data)}")
    print(f"Training: {len(train)}; held-out evaluation: {len(test)}")
    print(data.category.value_counts().to_string())


if __name__ == "__main__":
    prepare()
