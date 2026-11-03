# python-nlp

Small, self-contained NLP projects in Python. Each one has its own folder with
a README, data, code, a walkthrough notebook and generated outputs.

| Project | What it does | Techniques |
|---|---|---|
| [bbc-headlines-pos-ner](bbc-headlines-pos-ner/) | Finds what BBC News headlines are about and the language they use | Text cleaning, part-of-speech tagging, named entity recognition (spaCy, NLTK) |
| [sentiment-book-reviews](sentiment-book-reviews/) | Scores book review sentiment and compares it to star ratings | VADER, pre-trained transformer (Hugging Face) |
| [vectorization-bow-tfidf](vectorization-bow-tfidf/) | Turns text into numeric vectors and compares what each method captures | Bag of Words, TF-IDF (scikit-learn) |
| [topic-modelling-lda-lsa](topic-modelling-lda-lsa/) | Discovers the hidden themes in 100 news articles | Topic modelling: LDA, LSA, topic coherence (scikit-learn) |
| [text-classifier-logreg](text-classifier-logreg/) | Trains a sentiment classifier from 20 sentences and shows why that is too little data | Supervised classification, logistic regression, cross-validation (scikit-learn) |
| [fake-news-case-study](fake-news-case-study/) | Compares fake and factual news and asks what a classifier really learns | POS/NER, sentiment, TF-IDF classification, cross-validation (spaCy, scikit-learn) |

Each project lists its own dependencies in its `requirements.txt`, so you only install what you need.
