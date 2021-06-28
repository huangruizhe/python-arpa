from collections import OrderedDict
from collections import defaultdict
import math

from .base import ARPAModel
from .base import UNK


class Context(dict):
    """
    This class stores data for a context h.
    It behaves like a python dict object, except that it has several
    additional attributes.
    """
    def __init__(self):
        super().__init__()
        self.log_bo = None
        self.log_sum_p_seen = -math.inf

    def add_log_p(self, log_p, base):
        self.log_sum_p_seen = math.log(base ** log_p + base ** self.log_sum_p_seen, base)


class ARPAModelContext(ARPAModel):
    """
    This class stores an arpa language model grouped by context h.
    """
    def __init__(self, unk=UNK):
        super().__init__(unk=unk)
        self._counts = OrderedDict()

        self._ngrams = OrderedDict()  # Use self._ngrams[len(h)][h][w] for saving the entry of (h,w)
        self._vocabulary = set()

    def contains_word(self, word):
        self._check_word(word)
        return word in self._vocabulary

    def __contains__(self, ngram):
        h = ngram[:-1]  # h is a tuple
        w = ngram[-1]   # w is a string/word
        return h in self._ngrams[len(h)] and w in self._ngrams[len(h)][h]

    def add_count(self, order, count):
        self._counts[order] = count
        self._ngrams[order - 1] = defaultdict(Context)

    def add_entry(self, ngram, p, bo=None, order=None):
        # Note: ngram is a tuple of strings, e.g. ("w1", "w2", "w3")
        h = ngram[:-1]  # h is a tuple
        w = ngram[-1]   # w is a string/word

        # Note that p and bo here are in fact in the log domain (self._base = 10)
        h_context = self._ngrams[len(h)][h]
        h_context[w] = p
        h_context.add_log_p(p, self._base)
        if bo is not None:
            h_context.log_bo = bo

        for word in ngram:
            self._vocabulary.add(word)

    def counts(self):
        return sorted(self._counts.items())

    def order(self):
        return max(self._counts.keys(), default=None)

    def vocabulary(self, sort=True):
        if sort:
            return sorted(self._vocabulary)
        else:
            return self._vocabulary

    def _entries(self, order):
        return (self._entry(h, w) for h, wlist in self._ngrams[order - 1].items() for w in wlist)

    def _entry(self, h, w):
        # return the entry for the ngram (h, w)
        ngram = h + (w,)
        log_p = self._ngrams[len(h)][h][w]
        log_bo = self._log_bo(ngram)
        if log_bo is not None:
            return log_p, ngram, log_bo
        else:
            return log_p, ngram

    def _log_bo(self, ngram):
        if len(ngram) in self._ngrams and ngram in self._ngrams[len(ngram)]:
            return self._ngrams[len(ngram)][ngram].log_bo
        else:
            return None

    def _log_p(self, ngram):
        h = ngram[:-1]  # h is a tuple
        w = ngram[-1]   # w is a string/word
        if h in self._ngrams[len(h)] and w in self._ngrams[len(h)][h]:
            return self._ngrams[len(h)][h][w]
        else:
            return None

    def log_p_raw(self, ngram):
        log_p = self._log_p(ngram)
        if log_p is not None:
            return log_p
        else:
            if len(ngram) == 1:
                raise KeyError
            else:
                log_bo = self._log_bo(ngram[:-1])
                if log_bo is None:
                    log_bo = 0
                return log_bo + self.log_p_raw(ngram[1:])


