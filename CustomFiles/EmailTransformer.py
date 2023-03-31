from sklearn.base import BaseEstimator, TransformerMixin
import re
from html import unescape
import urlextract
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix


def html_to_plain_text(html):
    text = re.sub('<head.*?>.*?</head>', '', html, flags=re.M | re.S | re.I)
    text = re.sub('<a\s.*?>', ' HYPERLINK ', text, flags=re.M | re.S | re.I)
    text = re.sub('<.*?>', '', text, flags=re.M | re.S)
    text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)


def email_to_text(email):
    html = None
    for part in email.walk():
        ctype = part.get_content_type()
        if not ctype in ("text/plain", "text/html"):
            continue
        try:
            content = part.get_content()
        except: # in case of encoding issues
            content = str(part.get_payload())
        if ctype == "text/plain":
            return content
        else:
            html = content
    if html:
        return html_to_plain_text(html)


url_extractor = urlextract.URLExtract()


class EmailTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, lowercase=True, remove_punctuation=True, replace_urls=True, replace_numbers=True):

        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        X_transformed = []

        for email in X:

            text = email_to_text(email) or ""

            if self.lowercase:

                text = text.lower()

            if self.replace_urls:

                urls = list(set(url_extractor.find_urls(text)))
                urls.sort(key=lambda url: len(url), reverse=True)
                for url in urls:
                    text = text.replace(url, " URL ")

            if self.replace_numbers:

                text = re.sub(r'\d+(?:\.\d*(?:[eE]\d+))?', 'NUMBER', text)

            if self.remove_punctuation:

                text = re.sub(r'\W+', ' ', text, flags=re.M)

            word_counts = Counter(text.split())

            X_transformed.append(word_counts)

        return np.array(X_transformed)


class VectorTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, vocabulary_count=1000):

        self.vocabulary_count = vocabulary_count

    def fit(self, X, y=None):

        total_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                total_count[word] += min(count, 10)
        most_common = total_count.most_common()[:self.vocabulary_count]
        self.most_common_ = most_common
        self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(most_common)}
        return self

    def transform(self, X, y=None):

        rows = []
        cols = []
        data = []

        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)
        return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_count + 1))


