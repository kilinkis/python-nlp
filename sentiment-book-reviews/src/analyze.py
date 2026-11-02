"""Sentiment analysis of book reviews: VADER vs. a pre-trained transformer,
checked against the star ratings reviewers gave."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "book_reviews_sample.csv"
OUT = ROOT / "outputs"
MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
LABELS = ["negative", "neutral", "positive"]


def load_reviews(path: Path = DATA) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(subset=["reviewText", "rating"])
    df = df.drop_duplicates(subset="reviewText").reset_index(drop=True)
    # Star rating gives us a ground truth to compare the models against.
    df["rating_label"] = pd.cut(df["rating"], [0, 2, 3, 5], labels=LABELS)
    return df


def add_vader(df: pd.DataFrame) -> pd.DataFrame:
    """VADER runs on the raw text: it uses capitalisation, '!' and emoticons as signals."""
    vader = SentimentIntensityAnalyzer()
    df["vader_score"] = df["reviewText"].apply(lambda t: vader.polarity_scores(t)["compound"])
    df["vader_label"] = pd.cut(df["vader_score"], [-1, -0.05, 0.05, 1], labels=LABELS, include_lowest=True)
    return df


def add_transformer(df: pd.DataFrame) -> pd.DataFrame:
    """Binary positive/negative model; long reviews are truncated to the model's limit."""
    clf = pipeline("sentiment-analysis", model=MODEL)
    preds = clf(df["reviewText"].tolist(), truncation=True, batch_size=16)
    df["transformer_label"] = [p["label"].lower() for p in preds]
    df["transformer_score"] = [p["score"] for p in preds]
    return df


def evaluate(df: pd.DataFrame, pred_col: str, name: str) -> None:
    """Compare against star ratings. 3-star reviews are excluded: they have no clear polarity."""
    polar = df[df["rating_label"] != "neutral"]
    pred = polar[pred_col].astype(str)
    # VADER can answer "neutral" on a polar review; keep that column so the miss is visible.
    labels = LABELS if "neutral" in set(pred) else ["negative", "positive"]
    print(f"\n{name} vs. star ratings (1-2 = negative, 4-5 = positive, n={len(polar)})")
    print(classification_report(polar["rating_label"].astype(str), pred, labels=["negative", "positive"], zero_division=0))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        polar["rating_label"].astype(str), pred, labels=labels, ax=ax, colorbar=False
    )
    ax.set_title(f"{name} vs. star rating")
    ax.set_xlabel(f"{name} prediction")
    ax.set_ylabel("Star rating label")
    fig.tight_layout()
    fig.savefig(OUT / f"confusion_{name.lower()}.png", dpi=150)
    plt.close(fig)


def plot_distributions(df: pd.DataFrame) -> None:
    counts = pd.DataFrame({
        "Star rating": df["rating_label"].value_counts(),
        "VADER": df["vader_label"].value_counts(),
        "Transformer": df["transformer_label"].value_counts(),
    }).reindex(LABELS).fillna(0)
    ax = counts.plot.bar(figsize=(7, 4), rot=0)
    ax.set_title("Sentiment label distribution")
    ax.set_ylabel("Reviews")
    plt.tight_layout()
    plt.savefig(OUT / "label_distribution.png", dpi=150)
    plt.close()

    ax = df.groupby("rating")["vader_score"].mean().plot.bar(figsize=(6, 4), rot=0)
    ax.set_title("Mean VADER score by star rating")
    ax.set_xlabel("Star rating")
    ax.set_ylabel("Mean compound score")
    plt.tight_layout()
    plt.savefig(OUT / "vader_by_rating.png", dpi=150)
    plt.close()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = add_transformer(add_vader(load_reviews()))
    print(f"{len(df)} reviews")
    df.to_csv(OUT / "reviews_scored.csv", index=False)
    evaluate(df, "vader_label", "VADER")
    evaluate(df, "transformer_label", "Transformer")
    plot_distributions(df)


if __name__ == "__main__":
    main()
