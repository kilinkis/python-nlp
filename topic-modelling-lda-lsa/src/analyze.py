"""Topic modelling of news articles with LDA (probabilistic) and LSA (linear algebra)."""
import re
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "news_articles.csv"
OUT = ROOT / "outputs"
SEED = 0
TOP_WORDS = 10
TOPIC_RANGE = range(2, 11)
# Newswire boilerplate and filler speech that is frequent but says nothing about topic.
EXTRA_STOPWORDS = {"said", "say", "says", "mr", "ms", "mrs", "also", "would", "could", "one", "year", "time",
                   "going", "want", "something", "just", "like", "get", "know", "make", "even", "still"}


def load_articles(path: Path = DATA) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(subset=["content"])
    return df.drop_duplicates(subset="content").reset_index(drop=True)


def preprocess(text: str, stop: set, lemmatizer: WordNetLemmatizer) -> str:
    """Lowercase, keep alphabetic words of 3+ letters, drop stop words, lemmatize."""
    words = re.findall(r"[a-z]{3,}", text.lower())
    return " ".join(lemmatizer.lemmatize(w) for w in words if w not in stop)


def clean_articles(df: pd.DataFrame) -> pd.Series:
    for pkg in ("stopwords", "wordnet"):
        nltk.download(pkg, quiet=True)
    stop = set(stopwords.words("english")) | EXTRA_STOPWORDS
    lemmatizer = WordNetLemmatizer()
    return df["content"].apply(lambda t: preprocess(t, stop, lemmatizer))


def vectorize(texts: pd.Series):
    """Bag-of-words counts for LDA and the TF-IDF reweighting of them for LSA.

    Words in fewer than 3 articles or more than half of them are dropped:
    too rare to form a topic, or too common to separate topics.
    """
    counts_vec = CountVectorizer(min_df=3, max_df=0.5)
    counts = counts_vec.fit_transform(texts)
    tfidf = TfidfTransformer().fit_transform(counts)
    return counts, tfidf, counts_vec.get_feature_names_out()


def top_words(components: np.ndarray, vocab: np.ndarray, n: int = TOP_WORDS) -> list[list[str]]:
    # LSA components have arbitrary sign; rank by absolute weight.
    return [list(vocab[np.argsort(-np.abs(row))[:n]]) for row in components]


def npmi_coherence(topics: list[list[str]], counts, vocab: np.ndarray) -> float:
    """Average NPMI of every pair of top words, using co-occurrence within articles.

    Higher means a topic's words tend to appear in the same articles. Range -1 to 1.
    """
    present = (counts > 0).astype(float).tocsc()
    index = {w: i for i, w in enumerate(vocab)}
    n_docs = present.shape[0]
    scores = []
    for words in topics:
        pair_scores = []
        for a, b in combinations(words, 2):
            va, vb = present[:, index[a]], present[:, index[b]]
            p_a, p_b = va.sum() / n_docs, vb.sum() / n_docs
            p_ab = va.multiply(vb).sum() / n_docs
            if p_ab == 0:
                pair_scores.append(-1.0)
            else:
                pair_scores.append(np.log(p_ab / (p_a * p_b)) / -np.log(p_ab))
        scores.append(np.mean(pair_scores))
    return float(np.mean(scores))


def fit_lda(counts, k: int) -> LatentDirichletAllocation:
    return LatentDirichletAllocation(n_components=k, random_state=SEED, max_iter=50, learning_method="batch").fit(counts)


def fit_lsa(tfidf, k: int) -> TruncatedSVD:
    return TruncatedSVD(n_components=k, random_state=SEED).fit(tfidf)


def coherence_scan(counts, tfidf, vocab) -> pd.DataFrame:
    rows = []
    for k in TOPIC_RANGE:
        rows.append({
            "topics": k,
            "LDA": npmi_coherence(top_words(fit_lda(counts, k).components_, vocab), counts, vocab),
            "LSA": npmi_coherence(top_words(fit_lsa(tfidf, k).components_, vocab), counts, vocab),
        })
    return pd.DataFrame(rows).set_index("topics")


def plot_coherence(scan: pd.DataFrame) -> None:
    ax = scan.plot(marker="o", figsize=(7, 4))
    ax.set_title("Topic coherence by number of topics")
    ax.set_xlabel("Number of topics")
    ax.set_ylabel("Mean NPMI coherence")
    plt.tight_layout()
    plt.savefig(OUT / "coherence.png", dpi=150)
    plt.close()


def topic_table(topics: list[list[str]]) -> pd.DataFrame:
    return pd.DataFrame({f"topic {i + 1}": ws for i, ws in enumerate(topics)})


def dominant_topics(model, matrix, df: pd.DataFrame) -> pd.DataFrame:
    """Assign each article to its strongest topic."""
    weights = model.transform(matrix)
    if isinstance(model, TruncatedSVD):
        weights = np.abs(weights)
    out = df[["title"]].copy()
    out["topic"] = weights.argmax(axis=1) + 1
    return out


def plot_topic_sizes(lda_topics: pd.DataFrame, lsa_topics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, d, name in zip(axes, (lda_topics, lsa_topics), ("LDA", "LSA")):
        d["topic"].value_counts().sort_index().plot.bar(ax=ax, rot=0)
        ax.set_title(f"{name}: articles per dominant topic")
        ax.set_xlabel("Topic")
    plt.tight_layout()
    plt.savefig(OUT / "topic_sizes.png", dpi=150)
    plt.close()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = load_articles()
    texts = clean_articles(df)
    counts, tfidf, vocab = vectorize(texts)
    print(f"{len(df)} articles, {len(vocab)} vocabulary words")

    scan = coherence_scan(counts, tfidf, vocab)
    print("\nCoherence by number of topics:\n", scan.round(3))
    scan.round(4).to_csv(OUT / "coherence.csv")
    plot_coherence(scan)

    best = {m: int(scan[m].idxmax()) for m in scan}
    print("\nBest number of topics:", best)
    lda, lsa = fit_lda(counts, best["LDA"]), fit_lsa(tfidf, best["LSA"])
    lda_words, lsa_words = top_words(lda.components_, vocab), top_words(lsa.components_, vocab)
    for name, words in (("LDA", lda_words), ("LSA", lsa_words)):
        table = topic_table(words)
        print(f"\n{name} topics:\n{table.to_string()}")
        table.to_csv(OUT / f"{name.lower()}_topics.csv", index=False)

    lda_assign, lsa_assign = dominant_topics(lda, counts, df), dominant_topics(lsa, tfidf, df)
    lda_assign.to_csv(OUT / "lda_article_topics.csv", index=False)
    lsa_assign.to_csv(OUT / "lsa_article_topics.csv", index=False)
    plot_topic_sizes(lda_assign, lsa_assign)


if __name__ == "__main__":
    main()
