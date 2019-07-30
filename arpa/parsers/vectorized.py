from enum import Enum, unique
import re

from .quick import ARPAParserQuick
from ..exceptions import ParseException

import logging
logger = logging.getLogger("mdiLogger")


class ARPAParserVectorized(ARPAParserQuick):
    # Motivation: use less python's "for" loop
    # If you can use numpy's native functions, this is the most efficient way.

    def __init__(self, model):
        super().__init__(model)
        self.ModelClass = model

    def parse(self, fp):
        self._result = []
        self._state = self.State.DATA
        self._tmp_model = None
        self._tmp_order = None
        progress_count = 0
        for line in fp:
            line = line.strip()

            progress_count += 1
            if progress_count % 1000000 == 0:
                logger.info("loading arpa file: %d lines" % progress_count)

            if self._state == self.State.DATA:
                self._data(line)
            elif self._state == self.State.COUNT:
                self._count(line)
            elif self._state == self.State.HEADER:
                self._header(line)
            elif self._state == self.State.ENTRY:
                self._entry(line)
        if self._state != self.State.DATA:
            raise ParseException(line)
        return self._result

    def _count(self, line):
        match = self.re_count.match(line)
        if match:
            order = match.group(1)
            count = match.group(2)
            self._tmp_model.add_count(int(order), int(count))
            self._tmp_model.add_ngram_table(int(order), int(count))
        elif not line:
            self._state = self.State.HEADER  # there are no counts
        else:
            raise ParseException(line)

    def _header(self, line):
        match = self.re_header.match(line)
        if match:
            self._state = self.State.ENTRY
            self._tmp_order = int(match.group(1))
        elif line == "\\end\\":
            self._tmp_model.finalize()
            self._result.append(self._tmp_model)
            self._state = self.State.DATA
            self._tmp_model = None
            self._tmp_order = None
        elif not line:
            pass  # skip empty line
        else:
            raise ParseException(line)

    def _entry(self, line):
        match = self.re_entry.match(line)
        if match:
            p = match.group(1)
            ngram = tuple(match.group(4).split(" "))
            bo = match.group(7)
            self._tmp_model.add_entry(ngram, p, bo, self._tmp_order)
        elif not line:
            self._state = self.State.HEADER  # last entry
        else:
            raise ParseException(line)
