"""Turn text into numbers: Bag of Words (raw counts) vs. TF-IDF (weighted counts)."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "sentences.txt"
OUT = ROOT / "outputs"


def load_sentences(path: Path = DATA) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def vectorize(sentences: list[str], vectorizer) -> pd.DataFrame:
    """One row per sentence, one column per vocabulary word."""
    matrix = vectorizer.fit_transform(sentences)
    return pd.DataFrame(
        matrix.toarray(),
        columns=vectorizer.get_feature_names_out(),
        index=[f"S{i + 1}" for i in range(len(sentences))],
    )


def top_terms(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    rows = {
        idx: ", ".join(f"{w} ({v:.2f})" for w, v in row.nlargest(n).items())
        for idx, row in df.iterrows()
    }
    return pd.Series(rows, name="top terms").to_frame()


def plot_similarity(bow: pd.DataFrame, tfidf: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, df, name in zip(axes, (bow, tfidf), ("Bag of Words", "TF-IDF")):
        sim = cosine_similarity(df)
        im = ax.imshow(sim, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(df)), df.index)
        ax.set_yticks(range(len(df)), df.index)
        ax.set_title(f"Cosine similarity: {name}")
        for (i, j), v in pd.DataFrame(sim).stack().items():
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(OUT / "similarity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    sentences = load_sentences()
    bow = vectorize(sentences, CountVectorizer())
    tfidf = vectorize(sentences, TfidfVectorizer())
    print(f"{len(sentences)} sentences, {bow.shape[1]} vocabulary words")
    bow.to_csv(OUT / "bag_of_words.csv")
    tfidf.round(4).to_csv(OUT / "tfidf.csv")

    print("\nBag of Words, most frequent terms per sentence")
    print(top_terms(bow).to_string())
    print("\nTF-IDF, highest weighted terms per sentence")
    print(top_terms(tfidf).to_string())
    print("\nWord 'the': BoW count vs. TF-IDF weight")
    print(pd.DataFrame({"bow": bow["the"], "tfidf": tfidf["the"].round(3)}).to_string())
    plot_similarity(bow, tfidf)


if __name__ == "__main__":
    main()
