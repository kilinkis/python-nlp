"""Case study: how do fake and factual news differ, and can we tell them apart?"""
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "fake_news_data.csv"
OUT = ROOT / "outputs"
SEED = 0
LABELS = ["Factual News", "Fake News"]
COLOURS = {"Factual News": "#1f77b4", "Fake News": "#d62728"}

DATELINE = re.compile(r"^.{0,80}?\(Reuters\)\s*-?\s*")


def load_data(path: Path = DATA) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(subset=["text"])
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    df["text"] = df["text"].str.strip()
    df["words"] = df["text"].str.split().str.len()
    return df


# ---- exploration ------------------------------------------------------------

def tag_texts(df: pd.DataFrame, model: str = "en_core_web_sm") -> pd.DataFrame:
    """One row per document with POS-tag shares and the named entities found."""
    nlp = spacy.load(model, disable=["lemmatizer"])
    rows = []
    for doc in nlp.pipe(df["text"], batch_size=20):
        alpha = [t for t in doc if t.is_alpha]
        pos = Counter(t.pos_ for t in alpha)
        rows.append({
            **{f"pos_{p}": pos[p] / max(len(alpha), 1) for p in ("NOUN", "PROPN", "VERB", "ADJ", "ADV", "PRON")},
            "entities": [(e.text, e.label_) for e in doc.ents],
        })
    return pd.DataFrame(rows)


def pos_shares(df: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in tags if c.startswith("pos_")]
    out = tags[cols].groupby(df["fake_or_factual"]).mean().T
    out.index = [i.removeprefix("pos_") for i in out.index]
    return out


def top_entities(df: pd.DataFrame, tags: pd.DataFrame, label: str, kinds=("PERSON", "ORG", "GPE", "NORP"), n=10) -> pd.DataFrame:
    counts = Counter(e for ents in tags.loc[df["fake_or_factual"] == label, "entities"] for e in ents if e[1] in kinds)
    return pd.DataFrame([(t, k, c) for (t, k), c in counts.most_common(n)], columns=["entity", "type", "count"])


def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    vader = SentimentIntensityAnalyzer()
    df["sentiment"] = df["text"].apply(lambda t: vader.polarity_scores(t)["compound"])
    df["sentiment_label"] = pd.cut(df["sentiment"], [-1, -0.05, 0.05, 1], labels=["negative", "neutral", "positive"], include_lowest=True)
    return df


# ---- classification ---------------------------------------------------------

def strip_dateline(text: str) -> str:
    """Remove the 'CITY (Reuters) -' prefix every factual article starts with."""
    return DATELINE.sub("", text, count=1)


def strip_source(text: str) -> str:
    """Also remove any remaining mention of the news agency."""
    return re.sub(r"\bReuters\b", "", strip_dateline(text), flags=re.I)


VARIANTS = {
    "Raw text": lambda t: t,
    "Dateline removed": strip_dateline,
    "Dateline + source removed": strip_source,
}


def make_model(kind: str = "lr"):
    clf = LogisticRegression(max_iter=1000, random_state=SEED) if kind == "lr" else LinearSVC(random_state=SEED)
    return make_pipeline(TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True), clf)


def cv_splitter() -> RepeatedStratifiedKFold:
    return RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)


def compare_variants(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, fn in VARIANTS.items():
        X = df["text"].apply(fn)
        for kind, name in (("lr", "Logistic regression"), ("svm", "Linear SVM")):
            s = cross_val_score(make_model(kind), X, df["fake_or_factual"], cv=cv_splitter())
            rows.append({"text": variant, "model": name, "accuracy": s.mean(), "std": s.std()})
    return pd.DataFrame(rows)


def feature_weights(df: pd.DataFrame, fn, n: int = 12) -> pd.DataFrame:
    """Fit on everything and list the words pushing hardest towards each class."""
    model = make_model("lr").fit(df["text"].apply(fn), df["fake_or_factual"])
    vocab = model[0].get_feature_names_out()
    coefs = pd.Series(model[-1].coef_[0], index=vocab).sort_values()  # positive = Fake News
    return pd.DataFrame({
        "factual word": coefs.head(n).index, "factual weight": coefs.head(n).values,
        "fake word": coefs.tail(n)[::-1].index, "fake weight": coefs.tail(n)[::-1].values,
    })


def cv_confusion(df: pd.DataFrame, fn) -> np.ndarray:
    pred = cross_val_predict(make_model("lr"), df["text"].apply(fn), df["fake_or_factual"], cv=5)
    return pred


# ---- plots ------------------------------------------------------------------

def plot_pos(shares: pd.DataFrame) -> None:
    ax = (shares * 100).plot.bar(figsize=(7, 4), rot=0, color=[COLOURS[c] for c in shares.columns])
    ax.set_title("Share of words by part of speech")
    ax.set_ylabel("% of words")
    plt.tight_layout(); plt.savefig(OUT / "pos_shares.png", dpi=150); plt.close()


def plot_entities(factual: pd.DataFrame, fake: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, d, name in zip(axes, (factual, fake), LABELS):
        d = d.iloc[::-1]
        ax.barh(d["entity"], d["count"], color=COLOURS[name])
        ax.set_title(f"Top entities: {name}")
        ax.set_xlabel("Mentions")
    plt.tight_layout(); plt.savefig(OUT / "top_entities.png", dpi=150); plt.close()


def plot_sentiment(df: pd.DataFrame) -> None:
    share = pd.crosstab(df["fake_or_factual"], df["sentiment_label"], normalize="index") * 100
    ax = share.plot.bar(figsize=(6, 4), rot=0, color=["#d62728", "#999999", "#2ca02c"])
    ax.set_title("VADER sentiment by news type")
    ax.set_ylabel("% of articles"); ax.set_xlabel("")
    plt.tight_layout(); plt.savefig(OUT / "sentiment_by_class.png", dpi=150); plt.close()


def plot_variants(res: pd.DataFrame) -> None:
    pivot = res.pivot(index="text", columns="model", values="accuracy").reindex(list(VARIANTS))
    ax = (pivot * 100).plot.bar(figsize=(7, 4), rot=0)
    ax.axhline(50, color="grey", ls="--", lw=1)
    ax.set_ylim(0, 105); ax.set_ylabel("Cross-validated accuracy (%)"); ax.set_xlabel("")
    ax.set_title("Accuracy depends on what the model can see")
    plt.tight_layout(); plt.savefig(OUT / "accuracy_by_variant.png", dpi=150); plt.close()


def plot_weights(w: pd.DataFrame, title: str, filename: str) -> None:
    words = list(w["factual word"][::-1]) + list(w["fake word"])
    vals = list(w["factual weight"][::-1]) + list(w["fake weight"])
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.barh(words, vals, color=[COLOURS["Factual News"] if v < 0 else COLOURS["Fake News"] for v in vals])
    ax.set_title(title); ax.set_xlabel("← factual | fake →")
    fig.tight_layout(); fig.savefig(OUT / filename, dpi=150); plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = load_data()
    print(df["fake_or_factual"].value_counts().to_string())
    print("\nWords per article:\n", df.groupby("fake_or_factual")["words"].describe().round(0).to_string())

    tags = tag_texts(df)
    shares = pos_shares(df, tags)
    print("\nPOS shares:\n", (shares * 100).round(1).to_string())
    shares.round(4).to_csv(OUT / "pos_shares.csv"); plot_pos(shares)

    ents = {c: top_entities(df, tags, c) for c in LABELS}
    for c, e in ents.items():
        print(f"\nTop entities, {c}:\n", e.to_string(index=False))
        e.to_csv(OUT / f"entities_{c.split()[0].lower()}.csv", index=False)
    plot_entities(ents["Factual News"], ents["Fake News"])

    df = add_sentiment(df)
    print("\nSentiment:\n", pd.crosstab(df["fake_or_factual"], df["sentiment_label"], normalize="index").round(2).to_string())
    plot_sentiment(df)

    res = compare_variants(df)
    print("\nCross-validated accuracy:\n", res.round(3).to_string(index=False))
    res.round(4).to_csv(OUT / "model_comparison.csv", index=False); plot_variants(res)

    for name, fn, fname in (("Raw text", VARIANTS["Raw text"], "weights_raw.png"),
                            ("Dateline + source removed", strip_source, "weights_clean.png")):
        w = feature_weights(df, fn)
        print(f"\nTop words ({name}):\n", w.round(2).to_string(index=False))
        w.round(4).to_csv(OUT / fname.replace(".png", ".csv"), index=False)
        plot_weights(w, f"Top words: {name}", fname)

    pred = cv_confusion(df, strip_source)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(df["fake_or_factual"], pred, ax=ax, colorbar=False, xticks_rotation=20)
    ax.set_title("Cross-validated, source removed")
    fig.tight_layout(); fig.savefig(OUT / "confusion.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
