"""A custom text classifier: bag of words + logistic regression, built from 20 labelled sentences."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "sentences.csv"
OUT = ROOT / "outputs"
SEED = 7

NEW_SENTENCES = [
    "what a wonderful surprise, i am so happy",
    "this day has been absolutely terrible",
    "i lost my keys and missed the bus",
    "my team won the final and we celebrated all night",
    "i am not happy with how this went",  # negation: a bag of words can't see it
    "the meeting is at three o'clock",     # no sentiment words at all
]


def load_data(path: Path = DATA) -> pd.DataFrame:
    return pd.read_csv(path)


def make_model(vectorizer=None) -> "Pipeline":
    """Vectorizer and classifier in one pipeline, so the vocabulary is learned
    from training data only and never leaks in from the test set."""
    return make_pipeline(vectorizer or CountVectorizer(), LogisticRegression(random_state=SEED))


def split(df: pd.DataFrame, test_size: float = 0.3):
    return train_test_split(
        df["text"], df["sentiment"], test_size=test_size, random_state=SEED, stratify=df["sentiment"]
    )


def cross_validate(df: pd.DataFrame) -> pd.DataFrame:
    """With 20 sentences one train/test split is a coin flip; repeated k-fold is steadier."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=SEED)
    rows = []
    for name, vec in (("Bag of Words", CountVectorizer()), ("TF-IDF", TfidfVectorizer())):
        scores = cross_val_score(make_model(vec), df["text"], df["sentiment"], cv=cv)
        rows.append({"features": name, "mean accuracy": scores.mean(), "std": scores.std()})
    return pd.DataFrame(rows).set_index("features")


def word_weights(model, n: int = 10) -> pd.DataFrame:
    """Words with the largest positive/negative coefficients. A binary logistic
    regression treats the alphabetically later class (positive) as 1."""
    vocab = model[0].get_feature_names_out()
    coefs = pd.Series(model[-1].coef_[0], index=vocab).sort_values()
    return pd.concat(
        [coefs.head(n).rename("weight").reset_index().rename(columns={"index": "negative word"}),
         coefs.tail(n)[::-1].rename("weight").reset_index().rename(columns={"index": "positive word"})],
        axis=1,
    )


def plot_weights(model, n: int = 8) -> None:
    vocab = model[0].get_feature_names_out()
    coefs = pd.Series(model[-1].coef_[0], index=vocab).sort_values()
    top = pd.concat([coefs.head(n), coefs.tail(n)])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(top.index, top.values, color=["tab:red" if v < 0 else "tab:green" for v in top.values])
    ax.set_title("Words that push the prediction most")
    ax.set_xlabel("Logistic regression coefficient (← negative | positive →)")
    fig.tight_layout()
    fig.savefig(OUT / "word_weights.png", dpi=150)
    plt.close(fig)


def predict_new(model, sentences: list[str]) -> pd.DataFrame:
    proba = model.predict_proba(sentences)
    classes = list(model.classes_)
    return pd.DataFrame({
        "sentence": sentences,
        "prediction": model.predict(sentences),
        "confidence": proba.max(axis=1).round(2),
        "P(positive)": proba[:, classes.index("positive")].round(2),
    })


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = load_data()
    print(df["sentiment"].value_counts().to_string())

    X_train, X_test, y_train, y_test = split(df)
    model = make_model().fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"\nSingle split (train={len(X_train)}, test={len(X_test)})")
    print(classification_report(y_test, pred, zero_division=0))
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, colorbar=False)
    fig.tight_layout()
    fig.savefig(OUT / "confusion.png", dpi=150)
    plt.close(fig)

    cv = cross_validate(df)
    print("Repeated 5-fold cross-validation:\n", cv.round(3))
    cv.round(4).to_csv(OUT / "cross_validation.csv")

    final = make_model().fit(df["text"], df["sentiment"])
    weights = word_weights(final)
    print("\nStrongest words:\n", weights.round(2).to_string(index=False))
    weights.round(4).to_csv(OUT / "word_weights.csv", index=False)
    plot_weights(final)

    preds = predict_new(final, NEW_SENTENCES)
    print("\nNew sentences:\n", preds.to_string(index=False))
    preds.to_csv(OUT / "new_predictions.csv", index=False)


if __name__ == "__main__":
    main()
