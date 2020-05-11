from collections import OrderedDict
from collections import defaultdict
import math
import numpy as np
import pickle

from .base import ARPAModel
from .base import UNK, SOS, EOS

import logging
logger = logging.getLogger("mdiLogger")


class ARPAModelVectorized(ARPAModel):
    # Note:
    # all ngrams represented as tuples in interger ids, e.g. (1, 3, 2)

    class NgramTable:
        # This data structure can be easily converted to pandas data frames
        # pd.DataFrame(data=tb.tb)

        def __init__(self, order, count):
            self.count = count  # real count, not including the dummy rows
            self.order = order

            # TODO: knowing "count", we can initialize the elements in ngt as fixed-sized vector instead of lists

            # the following data structures must have the same sizes
            self.tb = dict({i: [] for i in range(order)})
            self.tb["logp"] = []
            self.tb["logbow"] = []

            self.tb["hidx"] = None  # history index
            self.tb["sidx"] = None  # suffix index

            # keep track of the row ids
            self.idx = dict()

            self.logp = self.tb["logp"]
            self.logbow = self.tb["logbow"]
            self.hidx = self.tb["hidx"]
            self.sidx = self.tb["sidx"]
            self.w = self.tb[order - 1]

        def __contains__(self, ngram):
            return ngram in self.idx

        def col(self, i):
            if type(i) == str:
                return self.tb[i]
            elif i >= 0:
                return self.tb[i]
            else:
                return self.tb[self.order + i]

        def cols(self):
            # return: col_label, col_array
            return self.tb.items()

        def ngrams(self):
            return self.idx.keys()

        def ith_entry(self, i):
            log10p = self.logp[i]
            log10bow = self.logbow[i]
            ngram = tuple([self.tb[j][i] for j in range(self.order)])
            if log10bow == math.nan or abs(log10bow) < -1e-4:
                return log10p, ngram
            else:
                return log10p, ngram, log10bow

        def entry(self, ngram):
            i = self.idx[ngram]
            log10p = self.logp[i]
            log10bow = self.logbow[i]
            if log10bow == math.nan or abs(log10bow) < -1e-4:
                return log10p, ngram
            else:
                return log10p, ngram, log10bow

        def entries(self):
            for i in range(self.count):
                yield self.ith_entry(i)

        def add_row(self, ngram, p, bow):
            # if len(ngram) != self._order:
            #     raise Exception

            rowid = len(self.tb["logp"])
            for i in range(self.order):
                self.tb[i].append(ngram[i])
            self.logp.append(p)
            self.logbow.append(bow)
            self.idx[ngram] = rowid

        def populate_hidx(self, ngt_lower):
            # example:
            # ngram: (w1, w2, w3)
            # h:     (w1, w3)
            # s:     (w2, w3)

            hidx = np.empty(self.count + 2, dtype=np.int32)  # two dummy rows
            sidx = np.empty(self.count + 2, dtype=np.int32)

            if ngt_lower is None:
                hidx.fill(0)
                sidx.fill(0)
            else:
                lower_idx = ngt_lower.idx
                for ngram, rowid in self.idx.items():  # for all ngrams in this ngt
                    h = ngram[:-1]  # history
                    s = ngram[1:]  # suffix

                    # Assumption: if this ngram hw is seen, then its history h
                    # and its suffix h'w are all seen.
                    # [Update] May 9 2020: this does not hold for pruned lm!!
                    # The history h must exist, to keep track of the backoff weights
                    # The suffix s might not exists, in this case, it
                    # [Solution] May 10 2020: when s does not exists, p(s) = 0,
                    # which means we do not need corrections.
                    # We may do the sanity check to see if the algorithm returns the
                    # same results as the naive algorithm
                    hidx[rowid] = lower_idx[h]
                    sidx[rowid] = lower_idx.get(s, -1)  # refer to the last row, which is row-length independent

                    # if s in lower_idx:
                    #     sidx[rowid] = lower_idx[s]
                    # else:
                    #     sidx[rowid] = -1  # refer to the last row, which is row-length independent

                # two dummy rows
                hidx[-2] = sidx[-2] = -2
                hidx[-1] = sidx[-1] = -1

            self.tb["hidx"] = hidx
            self.tb["sidx"] = sidx
            self.hidx = hidx
            self.sidx = sidx

        def validate(self):
            for i in range(self.order):
                if len(self.tb[i]) != self.count + 2:
                    raise Exception
            if len(self.tb["logp"]) != self.count + 2 \
                    or len(self.tb["logbow"]) != self.count + 2 \
                    or (self.order > 1 and len(self.tb["hidx"]) != self.count + 2) \
                    or (self.order > 1 and len(self.tb["sidx"]) != self.count + 2):
                raise Exception

        def is_empty(self):
            return len(self.tb[0]) == 0

        def __str__(self, without_log=False):
            ret = ""
            for ngram, rowid in self.idx.items():
                e = self.entry(ngram)
                p = 10 ** e[0] if without_log else e[0]
                if len(e) == 2:
                    ret += "%.6f\t%s\n" % (p, str(ngram))
                else:
                    bow = 10 ** e[2] if without_log else e[2]
                    ret += "%.6f\t%s\t%.6f\n" % (p, str(ngram), bow)
            return ret

    # for some technical reasons, we have to make it static
    _vocab = None

    FLOAT_NDIGITS = 7

    def num_round(self, num, ndigits=FLOAT_NDIGITS):
        return round(num, ndigits)

    def __init__(self, unk=UNK):
        super().__init__(unk=unk)
        self._counts = OrderedDict()
        self._ngts = dict()  # ngram tables
        self._vocab = ARPAModelVectorized._vocab
        self.log_ph = None

    def __contains__(self, ngram):
        if isinstance(ngram[0], str):
            ngram = self._vocab.integerize(ngram)
        l = len(ngram)
        return l in self._ngts and ngram in self._ngts[l]

    @classmethod
    def set_vocab(self, vocab):
        ARPAModelVectorized._vocab = vocab

    def add_count(self, order, count):
        self._counts[order] = count

    def add_entry(self, ngram, p, bo=None, order=None):
        ngram = tuple(self._vocab.w_id[w] for w in ngram)
        self._ngts[order].add_row(ngram, p, bo)

    def add_ngram_table(self, order, count):
        self._ngts[order] = self.NgramTable(order, count)

    # def _get_wid_and_populate(self, w):
    #     try:
    #         return self.w_id[w]
    #     # A try/except block is extremely efficient if no exceptions are raised.
    #     # Actually catching an exception is expensive.
    #     # However, we can expect only |V| times of exception
    #     except KeyError:
    #         new_id = len(self.w_id)
    #         self.w_id[w] = new_id
    #         self.id_w[new_id] = w
    #         return new_id

    def counts(self):
        return sorted(self._counts.items())

    def order(self):
        return max(self._counts.keys(), default=None)

    def vocabulary(self, sort=True):
        return self._vocab

    def _entries(self, order, deintegerized=True, rounded=True):
        # return (self._entry(k, deintegerized, rounded) for k in self._ngts[order].ngrams())

        ngt = self._ngts[order]
        for ngram, row_id in ngt.idx.items():
            ngram_w = ngram
            if deintegerized:
                ngram_w = self._vocab.deintegerize(ngram)
            logp = ngt.logp[row_id]
            logbow = ngt.logbow[row_id]

            if rounded:
                if np.isinf(logp):
                    logp = -99
                else:
                    logp = self.num_round(logp)

                if np.isnan(logbow) or logbow == 0.0:  # abs(logbow - 0) < 1e-7
                    yield logp, ngram_w
                else:
                    yield logp, ngram_w, logbow
            else:
                yield logp, ngram_w, logbow

    def _entry(self, ngram, deintegerized=True, rounded=True):
        ngram_id = ngram
        if not isinstance(ngram_id[0], int):
            ngram_id = self._vocab.integerize(ngram)
        e = self._ngts[len(ngram)].entry(ngram_id)

        e = list(e)

        if deintegerized:
            e[1] = self._vocab.deintegerize(e[1])

        if rounded:
            if np.isinf(e[0]):
                e[0] = -99
            else:
                e[0] = self.num_round(e[0])

            if len(e) == 3:
                logbow = e[2]
                # if np.isnan(logbow) or abs(logbow - 0) < 1e-7:  # no need to keep bow in the arpa
                if np.isnan(logbow) or logbow == 0.0:
                    e = e[:-1]
                else:
                    e[2] = self.num_round(logbow)

        return tuple(e)

    def integerize(self, ngram):
        return self._vocab.integerize(ngram)

    def deintegerize(self, ngram):
        return self._vocab.deintegerize(ngram)

    def _log_bo(self, ngram):
        if not isinstance(ngram[0], int):
            ngram = self._vocab.integerize(ngram)
        tb = self._ngts[len(ngram)]
        rowid = tb.idx[ngram]
        return tb.logbow[rowid]

    def _log_p(self, ngram):
        if not isinstance(ngram[0], int):
            ngram = self._vocab.integerize(ngram)
        tb = self._ngts[len(ngram)]
        rowid = tb.idx[ngram]
        return tb.logp[rowid]

    def log_p(self, ngram):
        if not isinstance(ngram[0], int):
            ngram = self._vocab.integerize(ngram)
        return self.log_p_raw(ngram)

    def log_p0(self, ngram):
        if not isinstance(ngram[0], int):
            ngram = self._vocab.integerize(ngram)
        return self.log_p_raw0(ngram)

    def log_p_raw(self, ngram):
        if not isinstance(ngram[0], int):
            ngram = self._vocab.integerize(ngram)
        order = len(ngram)
        row_id = self._ngts[order].idx.get(ngram, None)
        if row_id is not None:
            return self._ngts[order].logp[row_id]
        else:
            bo_id = self._ngts[order - 1].idx.get(ngram[:-1], None)
            if bo_id is None:  # there is no recursion for bow
                log_bo = 0
            else:
                log_bo = self._ngts[order - 1].logbow[bo_id]
            return float(log_bo) + self.log_p_raw(ngram[1:])

    # If I remember correctly, this is used only for lm_adapt, because it needs to save
    # the current lm_adapt as well as the lm_out
    def log_p_raw0(self, ngram):
        if not isinstance(ngram[0], int):
            ngram = self._vocab.integerize(ngram)
        order = len(ngram)
        row_id = self._ngts[order].idx.get(ngram, None)
        if row_id is not None:
            return self._ngts[order].logp0[row_id]
        else:
            bo_id = self._ngts[order - 1].idx.get(ngram[:-1], None)
            if bo_id is None:  # there is no recursion for bow
                log_bo = 0
            else:
                log_bo = self._ngts[order - 1].logbow0[bo_id]
            return float(log_bo) + self.log_p_raw0(ngram[1:])

    # def get_backoff_ps(self, s):
    #     for i in range(len(s), 0, -1):
    #         if s[-i:] in self.ngts[i].idx:
    #             j = self.ngts[i].idx[s]
    #             return self.ngts[i].p[j]
    #     raise Exception("get_backoff_ps(%s): invalid" % str(self.deintegerize(s)))

    def finalize(self):
        # add two dummy rows for each ngt, for future use (e.g. handling idx out of range error)
        for ngt_order, ngt in self._ngts.items():
            ngram = tuple([self._vocab.dummy] * ngt_order)
            ngt.add_row(ngram, "-99", "-99")
            ngt.add_row(ngram, "-99", "-99")
            del ngt.idx[ngram]

        # turn lists into np arrays
        for ngt_order, ngt in self._ngts.items():  # for each ngram table
            for i in range(ngt_order):
                ngt.tb[i] = np.asarray(ngt.tb[i], dtype=np.int32)
                if i == ngt_order - 1:
                    ngt.w = ngt.tb[i]

        # evaluate str literals, handling math.inf and math.nan
        for ngt_order, ngt in self._ngts.items():
            logp = np.asarray(ngt.logp, dtype=np.float64)
            logbow = np.asarray(ngt.logbow, dtype=np.float64)

            log_bow_zero = np.logical_and(np.isnan(logbow), ngt.w != self._vocab.eos)  # these bow are omitted in arpa
            np.place(logbow, log_bow_zero, 0)
            np.place(logbow, logbow <= -99, -math.inf)

            np.place(logp, logp <= -99, -math.inf)

            ngt.tb["logp"] = logp
            ngt.tb["logbow"] = logbow
            ngt.logp = logp
            ngt.logbow = logbow

        # populate hdix and sp
        for ngt_order, ngt in self._ngts.items():
            ngt.populate_hidx(self._ngts.get(ngt_order - 1, None))
            ngt.validate()

    # def w(self, order):
    #     return self._ngts[order].w
    #
    # def logp_col(self, order):
    #     return self._ngts[order].logp
    #
    # def logbow_col(self, order):
    #     return self._ngts[order].logbow
    #
    # def idx(self, order):
    #     return self._ngts[order].idx
    #
    # def hidx(self, order):
    #     return self._ngts[order].hidx
    #
    # def sidx(self, order):
    #     return self._ngts[order].sidx

    def col(self, order, i):
        if isinstance(i, int):
            return self._ngts[order].col(i)
        else:
            if i == "idx":
                return self._ngts[order].idx
            elif i == "w":
                return self._ngts[order].tb[order - 1]
            elif i in ["logp", "logbow", "hidx", "sidx"]:
                return self._ngts[order].tb[i]
            else:
                raise KeyError("col %d %s" % (order, str(i)))

    def cols(self, order):
        return {i: self.col(order, i) for i in range(order)}

    def ngrams(self, order):
        return self._ngts[order].ngrams()

    def index_of(self, ngram):
        if isinstance(ngram[0], str):
            ngram = self._vocab.integerize(ngram)
        return self._ngts[len(ngram)].idx[ngram]

    def set_log_ph(self, log_ph):
        self.log_ph = log_ph

    def is_seen_history(self, h):
        return h in self.log_ph

    def is_seen_ngram(self, hw):
        return hw in self._ngts[len(hw)]

    def get_ngt(self, order):
        return self._ngts[order]

    def dump_pkl(self, fname):
        assert fname.endswith('.pkl')
        with open(fname, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # compute ppl

    def log_p_raw_print(self, ngram):
        if not isinstance(ngram[0], int):
            ngram = self._vocab.integerize(ngram)
        order = len(ngram)
        row_id = self._ngts[order].idx.get(ngram, None)
        if row_id is not None:
            return order, self._ngts[order].logp[row_id]
        else:
            bo_id = self._ngts[order - 1].idx.get(ngram[:-1], None)
            if bo_id is None:  # there is no recursion for bow
                log_bo = 0
            else:
                log_bo = self._ngts[order - 1].logbow[bo_id]
            n, lp = self.log_p_raw_print(ngram[1:])
            return n, float(log_bo) + lp


    # def log_p_raw_print(self, ngram):
    #     if not isinstance(ngram[0], int):
    #         ngram = self._vocab.integerize(ngram)
    #     order = len(ngram)
    #     row_id = self._ngts[order].idx.get(ngram, None)
    #     if row_id is not None:
    #         return ngram, self._ngts[order].logp[row_id]
    #     else:
    #         bo_id = self._ngts[order - 1].idx.get(ngram[:-1], None)
    #         if bo_id is None:  # there is no recursion for bow
    #             log_bo = 0
    #         else:
    #             log_bo = self._ngts[order - 1].logbow[bo_id]
    #         ng, lp = self.log_p_raw_print(ngram[1:])
    #         return ng, float(log_bo) + lp

    def set_order(self):
        self.n = self.order()

    def _replace_unks(self, words):
        return tuple((w if w in self._vocab else self._unk) for w in words)

    def log_s(self, sentence, sos=SOS, eos=EOS):
        words = self._check_input(sentence)
        if self._unk:
            words = self._replace_unks(words)

        n_words = len(words)
        details = list()

        start = 1
        if sos:
            words = (sos, ) + words
            start = 2
        if eos:
            words = words + (eos, )

        result = 0
        for i in range(start, len(words) + 1):
            ngram = words[max(0, i-self.n): i]
            n, log_p = self.log_p_raw_print(ngram)

            details.append((ngram[-n:], log_p))

            result += log_p

        return result, n_words, details


class Vocabulary:

    def __init__(self, path, encoding="utf-8", tolower=False):
        self.w_id = dict()  # word -> word_id
        self.id_w = dict()  # word_id -> word

        self._load(path, encoding=encoding, tolower=tolower)

        self.dummy = len(self.w_id) + 1
        self.unk = self.w_id[UNK]
        self.eos = self.w_id[EOS]
        self.sos = self.w_id[SOS]

    def __contains__(self, word):
        return word in self.w_id

    def __iter__(self):
        return iter(self.id_w.keys())

    def __str__(self):
        ret = ""
        for w, id in self.w_id.items():
            ret += "%s -> %d\n" % (w, id)
        return ret

    def _load(self, path, encoding="utf-8", tolower=False):
        # Parse the vocabulary generated by SRILM options:
        # -write-vocab
        # or
        # -write-vocab-index
        with open(path, 'r', encoding=encoding) as fp:
            first_line = self.peek_line(fp).split(" ")
            if len(first_line) == 2:
                for line in fp:
                    line = line.strip()
                    if len(line) < 1:
                        continue
                    ll = line.split(" ")
                    wid = int(ll[0])
                    if tolower:
                        w = ll[1].lower()
                    else:
                        w = ll[1]
                    self.w_id[w] = wid
                    self.id_w[wid] = w
            else:
                for line in fp:
                    line = line.strip()
                    if len(line) < 1:
                        continue
                    if tolower:
                        w = line.lower()
                    else:
                        w = line
                    if w in self.w_id:
                        continue
                    new_id = len(self.w_id)
                    self.w_id[w] = new_id
                    self.id_w[new_id] = w
        for w in (UNK, EOS, SOS):
            if w not in self.w_id:
                new_id = len(self.w_id)
                self.w_id[w] = new_id
                self.id_w[new_id] = w

    def peek_line(self, fin):
        pos = fin.tell()
        line = fin.readline()
        fin.seek(pos)
        return line

    def integerize(self, ngram):
        if not isinstance(ngram[0], int):
            ngram = tuple(self.w_id[w] for w in ngram)
        return ngram

    def deintegerize(self, ngram):
        return tuple(self.id_w[wid] for wid in ngram)

    def check_type(self):
        for w, id in self.w_id.items():
            assert isinstance(w, str)
            assert isinstance(id, int)


# TODO: A more efficient way to implement this data structure may
# be to just store which group this ngram belong to.
# Pros: preprocessing to the most
# Cons: hard to debug, may lose the exact ngram information



# Test the performance of pandas dataframe constructor: pd.DataFrame(data=a)
# This is efficient as it does not copy data from inputs by default.
#
# import timeit
# setup = '''
# import pandas as pd
# import numpy as np
# n = %d
# a = dict({i: np.random.random(size=(n,)) for i in range(%d)})'''
# timeit.timeit("pd.DataFrame(data=a)", setup=setup % (1000000, 3), number=1)
#

# col_new = np.empty(len(col), dtype=np.uint32)


# pip3 uninstall arpa
# pip3 install --user -e /Users/huangruizhe/Downloads/PycharmProjects/python-arpa
# pip3 install --user --upgrade -e /Users/huangruizhe/Downloads/PycharmProjects/python-arpa

# https://stackoverflow.com/a/41537134/4563935
# pip3 install --user /export/fs04/a12/rhuang/lmadapt/scripts/20191011/python-arpa
# pip3 install --user --upgrade /export/fs04/a12/rhuang/lmadapt/scripts/20191011/python-arpa

# N = 20000000
# r = 400
# df = pd.DataFrame({
#     'A':np.random.randint(1, r, size=N),
#     'B':np.random.randint(1, r, size=N),
#     'C':np.random.randint(1, r, size=N),
#     'nume2':np.random.normal(0,1,N)})
# t3 = time.time() ;\
# a = df.groupby(by=['A','B'], sort=False, as_index=False).agg({"nume2": log10_sum_exp_array}) ;\
# t4 = time.time() ;\
# print(t4-t3, "seconds")
#



