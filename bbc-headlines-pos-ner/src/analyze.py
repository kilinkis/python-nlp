"""Linguistic analysis of BBC News headlines: cleaning, POS tagging and NER."""
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "bbc_news.csv"
OUT = ROOT / "outputs"
TOP_N = 10


def load_titles(path: Path = DATA) -> pd.Series:
    titles = pd.read_csv(path)["title"].dropna().astype(str)
    return titles.drop_duplicates().reset_index(drop=True)


def clean_titles(titles: pd.Series) -> pd.DataFrame:
    """Lowercase, drop stop words and punctuation, tokenize and lemmatize."""
    for pkg in ("stopwords", "wordnet", "punkt_tab"):
        nltk.download(pkg, quiet=True)
    stop = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    df = pd.DataFrame({"title": titles})
    df["lowercase"] = df["title"].str.lower()
    df["no_stopwords"] = df["lowercase"].apply(
        lambda t: " ".join(w for w in t.split() if w not in stop)
    )
    df["clean"] = df["no_stopwords"].apply(lambda t: re.sub(r"[^\w\s]", "", t))
    df["tokens_raw"] = df["title"].apply(word_tokenize)
    df["tokens_clean"] = df["clean"].apply(word_tokenize)
    df["lemmas"] = df["tokens_clean"].apply(
        lambda toks: [lemmatizer.lemmatize(t) for t in toks]
    )
    return df


def tag_titles(titles: pd.Series, model: str = "en_core_web_sm"):
    """Run spaCy on each headline separately so context never leaks between them.

    POS counts use lowercase lemmas of alphabetic tokens ("says"/"say" merge,
    punctuation is ignored); entities keep their original surface form.
    """
    nlp = spacy.load(model)
    pos_rows, ner_rows = [], []
    for doc in nlp.pipe(titles, batch_size=100):
        pos_rows += [(t.lemma_.lower(), t.pos_) for t in doc if t.is_alpha]
        ner_rows += [(e.text, e.label_) for e in doc.ents]
    pos = pd.DataFrame(pos_rows, columns=["lemma", "pos_tag"])
    ner = pd.DataFrame(ner_rows, columns=["entity", "ner_tag"])
    return pos, ner


def top_counts(df: pd.DataFrame, label_col: str, label: str, n: int = TOP_N) -> pd.DataFrame:
    text_col = df.columns[0]
    counts = Counter(df.loc[df[label_col] == label, text_col])
    return pd.DataFrame(counts.most_common(n), columns=[text_col, "count"])


def bar_chart(df: pd.DataFrame, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(df.iloc[::-1, 0], df.iloc[::-1, 1])
    ax.set_title(title)
    ax.set_xlabel("Occurrences")
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    titles = load_titles()
    print(f"{len(titles)} unique headlines")

    cleaned = clean_titles(titles)
    cleaned.to_csv(OUT / "cleaned_titles.csv", index=False)
    lemma_counts = Counter(sum(cleaned["lemmas"], [])).most_common(TOP_N)
    print("\nMost common lemmas:", lemma_counts)

    pos, ner = tag_titles(titles)
    results = {
        "nouns": top_counts(pos, "pos_tag", "NOUN"),
        "verbs": top_counts(pos, "pos_tag", "VERB"),
        "adjectives": top_counts(pos, "pos_tag", "ADJ"),
        "people": top_counts(ner, "ner_tag", "PERSON"),
        "places": top_counts(ner, "ner_tag", "GPE"),
        "organisations": top_counts(ner, "ner_tag", "ORG"),
    }
    for name, df in results.items():
        print(f"\nTop {name}:\n{df.to_string(index=False)}")
        df.to_csv(OUT / f"top_{name}.csv", index=False)
        bar_chart(df, f"Most common {name} in BBC headlines", f"top_{name}.png")


if __name__ == "__main__":
    main()
