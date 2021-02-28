from collections import OrderedDict, Counter
from sklearn.base import TransformerMixin
from sklearn.preprocessing import normalize
from typing import List, Union
import numpy as np


class BoW(TransformerMixin):
    """
    Bag of words tranformer class

    check out:
    https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html
    to know about TransformerMixin class
    """

    def __init__(self, k: int):
        """
        :param k: number of most frequent tokens to use
        """
        self.k = k
        # list of k most frequent tokens
        self.bow = None

    def fit(self, X: np.ndarray, y=None):
        """
        :param X: array of texts to be trained on
        """
        # task: find up to self.k most frequent tokens in texts_train,
        # sort them by number of occurences (highest first)
        # store most frequent tokens in self.bow

        splited = ' '.join(X).split()
        self.counter = Counter(splited)
        min_count = 1
        # tokens from token_counts keys that had at least min_count occurrences throughout the dataset
        self.tokens = [token for token, counts in self.counter.items() if counts >= min_count]
        # Add a special tokens for unknown and empty words
        token_len = min(len(self.tokens), self.k)

        # self.tokens = [UNK, PAD] + sorted(self.tokens)[:token_len]
        self.tokens = sorted(self.tokens)[:token_len]
        # token_len += 2
        self.k = token_len

        self.token_to_id = {token: idx for idx, token in enumerate(self.tokens)}
        self.bow = self.tokens

        # fit method must always return self
        return self


    def _text_to_bow(self, text: str) -> np.ndarray:
        """
        convert text string to an array of token counts. Use self.bow.
        :param text: text to be transformed
        :return bow_feature: feature vector, made by bag of words
        """

        result = np.zeros(len(self.bow))
        tokens = text.split()
        for ind, token in enumerate(self.bow):
            if token in tokens:
                result[ind] += tokens.count(token)

        # result = np.zeros(self.k)
        # for word in text:
        #   # if word in self.counter:
        #   if word in self.tokens:
        #       result[self.token_to_id[word]] += 1;
            # else:
            #   result[self.token_to_id["PAD"]] += 1;
          # else:
          #   result[self.token_to_id["UNK"]] += 1;

        # result = None
        # raise NotImplementedError
        return np.array(result, "float32")

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.bow is not None
        return np.stack([self._text_to_bow(text) for text in X])

    def get_vocabulary(self) -> Union[List[str], None]:
        return self.bow


class TfIdf(TransformerMixin):
    """
    Tf-Idf tranformer class
    if you have troubles implementing Tf-Idf, check out:
    https://streamsql.io/blog/tf-idf-from-scratch
    """

    def __init__(self, k: int = None, normalize: bool = False):
        """
        :param k: number of most frequent tokens to use
        if set k equals None, than all words in train must be considered
        :param normalize: if True, you must normalize each data sample
        after computing tf-idf features
        """
        self.k = k
        self.normalize = normalize

        # self.idf[term] = log(total # of documents / # of documents with term in it)
        self.idf = OrderedDict()

    def fit(self, X: np.ndarray, y=None):
        """
        :param X: array of texts to be trained on
        """
        # raise NotImplementedError
        splited = ' '.join(X).split()
        self.counter = Counter(splited)
        min_count = 1
        # tokens from token_counts keys that had at least min_count occurrences throughout the dataset
        self.tokens = [token for token, counts in self.counter.items() if counts >= min_count]
        # Add a special tokens for unknown and empty words
        token_len = min(len(self.tokens), self.k)

        # self.tokens = [UNK, PAD] + sorted(self.tokens)[:token_len]
        self.tokens = sorted(self.tokens)[:token_len]
        # token_len += 2
        self.k = token_len
        self.bow = self.tokens

        countDict = {}
        # Run through each review's tf dictionary and increment countDict's (word, doc) pair
        for review in X:
            for word, counts in Counter(review.split()).items():
                if word in countDict:
                    countDict[word] += 1
                else:
                    countDict[word] = 1

        self.idfDict = {}
        for word in countDict:
            self.idfDict[word] = np.log(len(X) / countDict[word])


        # fit method must always return self
        return self

    def _text_to_tf_idf(self, text: str) -> np.ndarray:
        """
        convert text string to an array tf-idfs.
        *Note* don't forget to normalize, when self.normalize == True
        :param text: text to be transformed
        :return tf_idf: tf-idf features
        """

        reviewTFDict = {}
        reviewTFIDFDict = {}
        for word in text.split():
            if word in reviewTFDict:
                reviewTFDict[word] += 1
            else:
                reviewTFDict[word] = 1
        for word in reviewTFDict:
            reviewTFDict[word] = reviewTFDict[word] / len(text.split())

        # for word in reviewTFDict:
        #
        #     reviewTFIDFDict[word] = reviewTFDict[word] * self.idfDict[word]

        result = np.zeros(len(self.bow))
        tokens = text.split()
        for ind, token in enumerate(self.bow):
            if token in tokens:
                result[ind] = reviewTFDict[token] * self.idfDict[token]
        if self.normalize:
            normalize(result.reshape(1, -1), norm = 'l2')
        # raise NotImplementedError
        return np.array(result, "float32")

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.idf is not None
        return np.stack([self._text_to_tf_idf(text) for text in X])
