import re
import math
from collections import Counter

# PART ONE
def clean(text):
    text = re.sub(r"https?://\S+", "", text)  # remove links
    text = re.sub(r"[^\w\s]", "", text)       # remove nonwords/whitespace
    text = re.sub(r"\s+", " ", text)          # collapse extra whitespace
    return text.lower().strip()

def remove_stopwords(doc_words, stopwords):
    essential_words = [word for word in doc_words if word not in stopwords]
    return essential_words

def stemming(doc_words):
    stems = [re.sub(r"(ing|ly|ment)$", "", word) for word in doc_words]
    return " ".join(stems)

def compute_tfidf(preprocessed, original_docs):
    # store all words per document
    all_docs_words = []
    for doc in preprocessed:
        with open(doc, "r") as f:
            words = f.read().split()
            all_docs_words.append(words)

    total_docs = len(all_docs_words)
    doc_freq = Counter()
    for words in all_docs_words:
        for w in set(words):
            doc_freq[w] += 1

    # compute IDF
    idf = {}
    for word in doc_freq:
        idf[word] = math.log(total_docs / doc_freq[word]) + 1

    # compute TF-IDF for each document
    for idx, words in enumerate(all_docs_words):
        counts = Counter(words)
        total = counts.total()

        tf = {word: counts[word] / total for word in counts}
        tfidf = {word: round(tf[word] * idf[word], 2) for word in tf}
        sorted_scores = sorted(tfidf.items(), key=lambda x: (-x[1], x[0]))
        top5 = sorted_scores[:5]

        original = original_docs[idx]
        outname = "tfidf_" + original
        with open(outname, "w") as f:
            f.write(str(top5))


def main():
    with open("tfidf_docs.txt", "r") as f:
        docs_list = [doc.strip() for doc in f]
    with open("stopwords.txt", "r") as f:
        stopwords = {word.strip() for word in f}

    preprocessed = []
    # PART ONE preprocessing per doc
    for doc in docs_list:
        with open(doc, "r") as f:
            text = f.read()

        text = clean(text)
        doc_words = text.split()
        doc_words = remove_stopwords(doc_words, stopwords)
        text = stemming(doc_words)
        with open("preproc_" + doc, "w") as f:
            f.write(text)
        preprocessed.append("preproc_" + doc)

    # PART TWO TFIDF
    compute_tfidf(preprocessed, docs_list)


if __name__ == "__main__":
    main()
