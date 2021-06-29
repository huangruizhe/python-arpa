from collections import OrderedDict
from collections import defaultdict
import math

from .base import ARPAModel
from .base import UNK
from .base import SOS
from .base import EOS


class Context(dict):
    """
    This class stores data for a context h.
    It behaves like a python dict object, except that it has several
    additional attributes.
    """
    def __init__(self):
        super().__init__()
        self.log_bo = None
        # self.log_sum_p_seen = -math.inf

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

    def update_counts(self):
        for order in range(1, self.order() + 1):
            count = sum([len(wlist) for _, wlist in self._ngrams[order - 1].items()])
            if count > 0:
                self._counts[order] = count

    def add_entry(self, ngram, p, bo=None, order=None):
        # Note: ngram is a tuple of strings, e.g. ("w1", "w2", "w3")
        h = ngram[:-1]  # h is a tuple
        w = ngram[-1]   # w is a string/word

        # Note that p and bo here are in fact in the log domain (self._base = 10)
        h_context = self._ngrams[len(h)][h]
        h_context[w] = p
        if bo is not None:
            self._ngrams[len(ngram)][ngram].log_bo = bo

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

    def log_joint_prob(self, sequence):
        # Compute the joint prob of the sequence based on the chain rule
        # Note that sequence should be a tuple of strings
        #
        # Reference:
        # https://github.com/BitSpeech/SRILM/blob/d571a4424fb0cf08b29fbfccfddd092ea969eae3/lm/src/LM.cc#L527

        log_joint_p = 0
        seq = sequence
        while len(seq) > 0:
            log_joint_p += self.log_p_raw(seq)
            seq = seq[:-1]

            # If we're computing the marginal probability of the unigram
            # <s> context we have to look up </s> instead since the former
            # has prob = 0.
            if len(seq) == 1 and seq[0] == SOS:
                seq = (EOS,)

        return log_joint_p

    def set_new_context(self, h):
        old_context = self._ngrams[len(h)][h]
        self._ngrams[len(h)][h] = Context()
        return old_context
