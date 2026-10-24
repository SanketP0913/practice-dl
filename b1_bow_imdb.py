# B1 - text classification with bag of words (IMDB movie reviews)
#
# Starting NLP. The task is to read a movie review and say if it's positive or
# negative. The data is the IMDB dataset from Stanford: 50k real reviews, 25k for
# training and 25k for testing, half positive and half negative.
#
# "Bag of words" means i ignore word order completely. Each review just becomes
# counts of which words show up in it. I want to see how far that gets before
# trying embeddings (B2) and a transformer (B3).
#
# What i compare:
#   guessing       - always say positive (50%)
#   word list      - my own rule: count "good" words vs "bad" words
#   naive bayes    - word counts + probabilities, the classic spam filter
#   logreg words   - logistic regression on tf-idf of single words
#   logreg +pairs  - same but also with word pairs (bigrams), so "not good" becomes one feature
#
# first run downloads the data (~85 MB) and saves it as imdb.csv so it's faster next time.
#
# pip install scikit-learn pandas pyarrow
# python b1_bow_imdb.py
# python b1_bow_imdb.py --try "the acting was not good at all"

import argparse
import csv
import io
import os
import re
import tarfile
import time
import urllib.request

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# stanford's own server was down for me, so i get the same data from the hugging face copy
HF_URL = "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/plain_text/{}-00000-of-00001.parquet"
STANFORD_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
CACHE = "imdb.csv"
csv.field_size_limit(10_000_000)

# my hand-made word lists for the rule baseline
GOOD = {"good", "great", "excellent", "best", "love", "loved", "wonderful", "amazing",
        "perfect", "brilliant", "fantastic", "enjoyed", "beautiful", "favorite", "fun"}
BAD = {"bad", "worst", "awful", "boring", "terrible", "waste", "poor", "horrible",
       "stupid", "worse", "dull", "lame", "mess", "disappointing", "ridiculous"}


def clean(text):
    text = text.replace("<br />", " ")   # the reviews have html line breaks in them
    return text.lower()


def rows_from_hf():
    import pandas as pd  # needs pandas + pyarrow to read parquet files
    rows = []
    for split in ("train", "test"):
        df = pd.read_parquet(HF_URL.format(split))
        rows += [(split, int(l), t) for t, l in zip(df["text"], df["label"])]
    return rows


def rows_from_stanford():
    data = urllib.request.urlopen(STANFORD_URL).read()
    rows = []
    # file names look like aclImdb/train/pos/123_9.txt
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for m in tar:
            parts = m.name.split("/")
            if len(parts) == 4 and parts[1] in ("train", "test") and parts[2] in ("pos", "neg") \
                    and m.name.endswith(".txt"):
                text = tar.extractfile(m).read().decode("utf-8")
                rows.append((parts[1], 1 if parts[2] == "pos" else 0, text))
    return rows


def download_imdb():
    print("downloading IMDB reviews, only happens once...")
    try:
        rows = rows_from_hf()
    except (OSError, ValueError, ImportError) as e:   # download failed or pandas/pyarrow missing
        print(f"hugging face didn't work ({e}), trying stanford")
        rows = rows_from_stanford()
    with open(CACHE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "label", "text"])
        w.writerows(rows)
    print(f"saved {len(rows):,} reviews to {CACHE}")


def load_imdb():
    if not os.path.exists(CACHE):
        download_imdb()
    data = {"train": ([], []), "test": ([], [])}
    with open(CACHE) as f:
        for row in csv.DictReader(f):
            texts, labels = data[row["split"]]
            texts.append(clean(row["text"]))
            labels.append(int(row["label"]))
    (xtr, ytr), (xte, yte) = data["train"], data["test"]
    return xtr, np.array(ytr), xte, np.array(yte)


def word_list_rule(texts):
    preds = []
    for t in texts:
        words = re.findall(r"[a-z']+", t)
        good = sum(w in GOOD for w in words)
        bad = sum(w in BAD for w in words)
        preds.append(1 if good >= bad else 0)
    return np.array(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--try", dest="tries", nargs="*", default=[],
                    help="your own sentences to test at the end")
    args = ap.parse_args()

    xtr, ytr, xte, yte = load_imdb()
    print(f"train {len(xtr):,} reviews ({ytr.mean():.0%} positive), test {len(xte):,}")
    print(f"average review length: {np.mean([len(t.split()) for t in xtr]):.0f} words")

    results = {}
    results["guessing (always positive)"] = (yte == 1).mean()
    results["my word list rule"] = (word_list_rule(xte) == yte).mean()

    # naive bayes uses raw counts
    t0 = time.time()
    counts = CountVectorizer(min_df=2)
    nb = MultinomialNB().fit(counts.fit_transform(xtr), ytr)
    results["naive bayes"] = (nb.predict(counts.transform(xte)) == yte).mean()
    print(f"\nnaive bayes: {len(counts.vocabulary_):,} words in vocab ({time.time() - t0:.0f}s)")

    # tf-idf = word count, but words that show up in every review ("the", "movie") count less
    models = {}
    for name, ngrams in [("logreg words", (1, 1)), ("logreg words + pairs", (1, 2))]:
        t0 = time.time()
        vec = TfidfVectorizer(ngram_range=ngrams, min_df=2, max_df=0.9, sublinear_tf=True)
        x = vec.fit_transform(xtr)
        clf = LogisticRegression(C=5, max_iter=2000).fit(x, ytr)
        results[name] = (clf.predict(vec.transform(xte)) == yte).mean()
        models[name] = (vec, clf)
        print(f"{name}: {x.shape[1]:,} features ({time.time() - t0:.0f}s)")

    print(f"\ntest accuracy ({len(xte):,} reviews):")
    for name, acc in results.items():
        print(f"  {name:28s} {acc:6.1%}")

    # what did the best model learn? biggest weights = most positive / negative words
    vec, clf = models["logreg words + pairs"]
    vocab = np.array(vec.get_feature_names_out())
    w = clf.coef_[0]
    order = np.argsort(w)
    print("\nmost negative words/pairs:", ", ".join(vocab[order[:15]]))
    print("most positive words/pairs:", ", ".join(vocab[order[-15:][::-1]]))

    # look at how it handles "not"
    print("\nweights for some pairs with 'not':")
    for p in ["good", "bad", "not good", "not bad", "great", "not great", "funny", "not funny"]:
        if p in vec.vocabulary_:
            print(f"  {p:10s} {w[vec.vocabulary_[p]]:+.2f}")

    # some mistakes, so i can see what's hard
    preds = clf.predict(vec.transform(xte))
    wrong = np.where(preds != yte)[0]
    rng = np.random.default_rng(0)
    print(f"\nit got {len(wrong):,} wrong. a few of them:")
    for i in rng.choice(wrong, min(3, len(wrong)), replace=False):
        label = "positive" if yte[i] else "negative"
        print(f"\n  [actually {label}] {xte[i][:300]}...")

    # my own sentences
    single_vec, single_clf = models["logreg words"]
    tries = args.tries or ["this movie was not good", "this movie was not bad at all",
                           "i expected it to be boring but it was great",
                           "great actors, terrible script"]
    print("\nmy own sentences (P(positive) from words only vs words + pairs):")
    for s in tries:
        p1 = single_clf.predict_proba(single_vec.transform([clean(s)]))[0, 1]
        p2 = clf.predict_proba(vec.transform([clean(s)]))[0, 1]
        print(f"  {p1:.2f}  {p2:.2f}   {s}")


if __name__ == "__main__":
    main()
