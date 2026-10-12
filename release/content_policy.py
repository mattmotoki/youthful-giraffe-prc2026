"""One bounded incremental policy for organiser movement records.

Positions are decoded character positions. A complete key/detail token must fit
inside [ID.start - WINDOW, ID.end + WINDOW], endpoints inclusive. Each ID has
its own relation; unrelated IDs cannot donate half of a relation. Keyed values
also match the bounded separator grammar below. Integer and zero-decimal forms
have one canonical spelling. No candidate is accepted at an artificial cut.
"""
from __future__ import annotations

import re

MOVEMENT_INTEGER = r"(?:1[89]|2[01])[0-9]{7}"
MOVEMENT_NUMBER = rf"{MOVEMENT_INTEGER}(?:\.0+)?"
MOVEMENT_RELATION_WINDOW = 512
MOVEMENT_FIELD_NAMES = ("MVT_ID", "MVT_ID_mvt", "mvt_id", "row_id", "row_ids",
                        "forced_in", "forced_out")
MOVEMENT_KEY_NAMES = MOVEMENT_FIELD_NAMES + ("id",)
MOVEMENT_DETAIL_NAMES = (
    "target", "y", "y_true", "TAXITIME_SEC_mvt", "taxitime_sec_mvt",
    "callsign", "flight", "FLIGHT_mvt", "airport", "ADEP_mvt", "stand",
    "STAND_mvt", "runway", "RUNWAY_mvt", "aircraft_type",
    "AIRCRAFT_TYPE_mvt", "recon",
)
MOVEMENT_ID_MIN = 3
MOVEMENT_FREE_TEXT_ALLOWED = frozenset({
    "experiments/exp-066-sparseholdout-preflight/tests/test_preflight.py",
    "experiments/exp-068-runway-queue/tests/test_queue_ceiling.py",
})
ID_COLUMNS = frozenset({"MVT_ID_mvt", "MVT_ID", "mvt_id"})
TARGET_COLUMNS = frozenset({"y", "y_true", "target", "TAXITIME_SEC_mvt", "taxitime_sec_mvt"})
DETAIL_COLUMNS = frozenset({"FLIGHT_mvt", "STAND_mvt", "RUNWAY_mvt",
                            "AIRCRAFT_TYPE_mvt", "callsign", "recon"})


def token_pattern(names):
    return r"(?<!\w)(?:" + "|".join(map(re.escape, names)) + r")(?!\w)"


MOVEMENT_ID = re.compile(rf"(?<![\w.#])({MOVEMENT_NUMBER})(?![\w.#])")
MOVEMENT_FIELD = re.compile(token_pattern(MOVEMENT_FIELD_NAMES), re.I)
MOVEMENT_KEY = re.compile(token_pattern(MOVEMENT_KEY_NAMES), re.I)
MOVEMENT_DETAIL = re.compile(token_pattern(MOVEMENT_DETAIL_NAMES), re.I)
KEYED_GAP_LIMIT = 16
KEYED_SEPARATOR = (rf"[\s\"'`]{{0,{KEYED_GAP_LIMIT}}}[:=|\[]"
                   rf"[\s\"'`\[]{{0,{KEYED_GAP_LIMIT}}}-?")
KEYED_PREFIX = re.compile(token_pattern(MOVEMENT_FIELD_NAMES) + KEYED_SEPARATOR + r"\Z", re.I)
KEYED_MOVEMENT = re.compile(token_pattern(MOVEMENT_FIELD_NAMES) + KEYED_SEPARATOR +
                            rf"{MOVEMENT_NUMBER}(?![\w.#])", re.I)
MAX_NAME = max(map(len, MOVEMENT_KEY_NAMES + MOVEMENT_DETAIL_NAMES))
# Include one actual left boundary; a suffix of an overlong word cannot be a key.
PREFIX_LIMIT = max(map(len, MOVEMENT_FIELD_NAMES)) + 2 * KEYED_GAP_LIMIT + 3
RUNS = re.compile(r"\w+|[.#]+|[^\w.#]+")
WORD = re.compile(r"\w")
ASCII_DIGITS = re.compile(r"[0-9]+\Z")
ZEROES = re.compile(r"0+\Z")


def canonical_movement_id(value: str | int) -> str | None:
    text = str(value)
    return text.split(".", 1)[0] if re.fullmatch(MOVEMENT_NUMBER, text) else None


class MovementScanner:
    """feed(str), then finalize(); repeated finalize returns the same result.

    feed after finalize raises ValueError, including an empty feed. Retained
    state has no absolute position, token length, or event log. All ages saturate
    at WINDOW+1. At most three pending relation masks survive: for equal masks,
    the latest ID end dominates every older one. The current ID snapshots its
    left relation before a zero suffix of any length. State retains at most
    PREFIX_LIMIT + MAX_NAME + 9 + 9*ID_MIN characters, plus a bounded header
    cell and finite column set; see state_bound() for inspectable limits.
    Input chunks and temporary regex matches belong to the caller's read bound.
    """
    def __init__(self, rel="fixture.txt", suffix=""):
        self.free_allowed = rel in MOVEMENT_FREE_TEXT_ALLOWED
        self.ids = set()
        self.field = False
        self.structured = False
        self.closed = False
        self.word = ""
        self.word_overlong = False
        self.atom = False
        self.number = ""
        self.number_phase = "integer"
        self.left_mask = 0
        self.keyed = False
        self.past = [MOVEMENT_RELATION_WINDOW + 1] * 2
        self.pending = {}  # mask -> age since ID end; masks: key=1, detail=2
        self.tail = ""
        self.header_open = suffix.lower() in (".csv", ".tsv")
        self.delimiter = "\t" if suffix.lower() == ".tsv" else ","
        self.cell = ""
        self.cell_bad = False
        self.columns = set()

    @staticmethod
    def state_bound():
        return {"prefix_characters": PREFIX_LIMIT, "word_characters": MAX_NAME,
                "integer_characters": 9, "distinct_ids": MOVEMENT_ID_MIN,
                "pending_relations": 3, "past_ages": 2,
                "maximum_age": MOVEMENT_RELATION_WINDOW + 1,
                "header_cell_characters": MAX_NAME + 4,
                "header_columns": len(ID_COLUMNS | TARGET_COLUMNS | DETAIL_COLUMNS)}

    def _advance(self, count):
        cap = MOVEMENT_RELATION_WINDOW + 1
        self.past = [min(cap, age + count) for age in self.past]
        self.pending = {mask: age + count for mask, age in self.pending.items()
                        if age + count <= MOVEMENT_RELATION_WINDOW}

    def _event(self, mask, length):
        for index, bit in enumerate((1, 2)):
            if mask & bit:
                self.past[index] = length
        updated = {}
        for old, age in self.pending.items():
            new = old | mask
            if new == 3:
                self.structured = True
            else:
                updated[new] = min(age, updated.get(new, age))
        self.pending = updated

    def _end_word(self):
        if self.word and not self.word_overlong:
            self.field |= bool(MOVEMENT_FIELD.fullmatch(self.word))
            mask = (1 if MOVEMENT_KEY.fullmatch(self.word) else 0) | (2 if MOVEMENT_DETAIL.fullmatch(self.word) else 0)
            if mask:
                self._event(mask, len(self.word))
        self.word = ""
        self.word_overlong = False

    def _end_atom(self):
        if self.atom and self.number_phase in ("integer", "zero"):
            if canonical_movement_id(self.number) is not None:
                if len(self.ids) < MOVEMENT_ID_MIN:
                    self.ids.add(self.number)
                if self.keyed or self.left_mask == 3:
                    self.structured = True
                else:
                    self.pending[self.left_mask] = 0
        self.atom = False
        self.number = ""
        self.number_phase = "integer"

    def _number_part(self, value, word):
        if self.number_phase == "bad":
            return
        if word:
            if self.number_phase == "integer":
                if len(self.number) + len(value) <= 9 and ASCII_DIGITS.fullmatch(value):
                    self.number += value
                else:
                    self.number_phase = "bad"
            elif self.number_phase in ("dot", "zero") and ZEROES.fullmatch(value):
                self.number_phase = "zero"
            else:
                self.number_phase = "bad"
        elif value == "." and self.number_phase == "integer" and len(self.number) == 9:
            self.number_phase = "dot"
        else:
            self.number_phase = "bad"

    def _header(self, chunk):
        if not self.header_open:
            return
        # Collapse runs of whitespace/quotes, but never remove internal spaces.
        # This implements strip().strip('"') without storing an unbounded cell.
        pattern = r'[^' + re.escape(self.delimiter) + r'\n]+|[' + re.escape(self.delimiter) + r'\n]'
        for match in re.finditer(pattern, chunk):
            part = match.group()
            if part in (self.delimiter, "\n"):
                self._end_cell()
                if part == "\n":
                    self.header_open = False
                    break
            elif not self.cell_bad:
                for item in re.finditer(r'\s+|"+|[^\s"]+', part):
                    value = item.group()
                    value = " " if value.isspace() else '"' if value[0] == '"' else value
                    if value in (' ', '"') and self.cell.endswith(value):
                        continue
                    if len(self.cell) + len(value) > MAX_NAME + 4:
                        self.cell_bad = True
                        self.cell = ""
                        break
                    self.cell += value

    def _end_cell(self):
        if not self.cell_bad:
            name = self.cell.strip().strip('"')
            if name in ID_COLUMNS | TARGET_COLUMNS | DETAIL_COLUMNS:
                self.columns.add(name)
        self.cell = ""
        self.cell_bad = False

    def feed(self, chunk: str):
        if self.closed:
            raise ValueError("movement scanner is finalized")
        self._header(chunk)
        for match in RUNS.finditer(chunk):
            value = match.group()
            word = bool(WORD.match(value))
            atom = word or value[0] in ".#"
            if not word:
                self._end_word()
            if not atom:
                self._end_atom()
            elif not self.atom:
                self.atom = True
                self.left_mask = sum(bit for bit, age in zip((1, 2), self.past)
                                     if age <= MOVEMENT_RELATION_WINDOW)
                self.keyed = bool(KEYED_PREFIX.search(self.tail))
            if word and not self.word_overlong:
                if len(self.word) + len(value) <= MAX_NAME:
                    self.word += value
                else:
                    self.word = ""
                    self.word_overlong = True
            if atom:
                self._number_part(value, word)
            self._advance(len(value))
            self.tail = (self.tail + value[-PREFIX_LIMIT:])[-PREFIX_LIMIT:]

    def finalize(self):
        if not self.closed:
            self._end_word()
            self._end_atom()
            if self.header_open:
                self._end_cell()
            self.closed = True
        ids = self.columns & ID_COLUMNS
        targets = self.columns & TARGET_COLUMNS
        table = sorted(ids | targets | (self.columns & DETAIL_COLUMNS if ids else set()))
        return {"ids": sorted(self.ids), "field": self.field,
                "structured": self.structured,
                "free": not self.free_allowed and self.field and len(self.ids) >= MOVEMENT_ID_MIN,
                "columns": table}

    def findings(self, rel):
        result = self.finalize()
        hits = []
        if result["columns"]:
            hits.append({"path": rel, "columns": result["columns"], "shape": "table"})
        if result["free"]:
            hits.append({"path": rel, "ids": len(result["ids"]), "shape": "free"})
        if result["structured"]:
            hits.append({"path": rel, "shape": "structured"})
        return hits


def movement_ids(text: str) -> set[str]:
    """Return the policy's bounded distinct-ID sample (saturates at ID_MIN)."""
    scanner = MovementScanner()
    scanner.feed(text)
    return set(scanner.finalize()["ids"])


def has_structured_movement(text: str) -> bool:
    scanner = MovementScanner()
    scanner.feed(text)
    return scanner.finalize()["structured"]
