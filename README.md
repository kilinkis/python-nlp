# python-nlp

Small, self-contained NLP projects in Python. Each one has its own folder with
a README, data, code, a walkthrough notebook and generated outputs.

| Project | What it does | Techniques |
|---|---|---|
| [bbc-headlines-pos-ner](bbc-headlines-pos-ner/) | Finds what BBC News headlines are about and the language they use | Text cleaning, part-of-speech tagging, named entity recognition (spaCy, NLTK) |
| [sentiment-book-reviews](sentiment-book-reviews/) | Scores book review sentiment and compares it to star ratings | VADER, pre-trained transformer (Hugging Face) |
| [vectorization-bow-tfidf](vectorization-bow-tfidf/) | Turns text into numeric vectors and compares what each method captures | Bag of Words, TF-IDF (scikit-learn) |

Each project lists its own dependencies in its `requirements.txt`, so you only install what you need.
