#!/usr/bin/env python3
"""Deterministic inference pipeline for the BRACIS 2026 Jusbrasil challenge.

The predictor intentionally never reads ``goldenset_offsets.csv``.  It extracts legal
references from the original Unicode text, resolves explicit identifiers
against the frozen SQLite database, and emits both contract JSON and Kaggle
CSV files.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import sqlite3
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Iterable

from canonical_metadata import CanonicalMetadataIndex

ROOT = Path(__file__).resolve().parent

COURTS = (
    "ac|al|ap|am|ba|ce|df|es|go|ma|mt|ms|mg|pa|pb|pr|pe|pi|rj|rn|rs|ro|rr|sc|sp|se|to"
)
NUMBER_BODY = r"[0-9](?:[0-9olsg.\s/\-]*[0-9olsg])?"
NUMBER_MARKER = r"(?:\s+n(?:[.o]\s*)?)?"
STATE_SUFFIX = rf"(?:\s*(?:/|-)\s*(?:{COURTS})|\s*\(\s*(?:{COURTS})\s*\))?"
BODY_DELIMITER_RE = re.compile(r"\n\s*\n\s*\n")
BODY_HEADING_RE = re.compile(
    r"^[ \t]*(?:relatorio|fundamentacao|ementa|voto|do\s+merito|merito)[ \t]*\r?$",
    re.IGNORECASE | re.MULTILINE,
)


def body_start(text: str) -> int:
    """Return a conservative argumentative-body offset from document structure."""
    delimiter = BODY_DELIMITER_RE.search(text)
    if delimiter:
        return delimiter.end()

    view = matching_view(text)
    for heading in BODY_HEADING_RE.finditer(view, 0, min(len(view), 5000)):
        header = view[: heading.start()]
        metadata_cues = sum(
            bool(re.search(pattern, header))
            for pattern in (
                r"\bprocesso\b",
                r"\brelator\b",
                r"\btribunal\b",
                r"\bclasse\s+processual\b",
            )
        )
        if metadata_cues >= 2:
            return heading.end()
    return 0


def matching_view(text: str) -> str:
    """Return a one-codepoint-per-codepoint normalized view for regex matching."""
    replacements = {
        "\u00a0": " ",
        "–": "-",
        "—": "-",
        "−": "-",
        "º": "o",
        "°": "o",
    }
    out: list[str] = []
    for char in text:
        if char in replacements:
            out.append(replacements[char])
            continue
        decomposed = unicodedata.normalize("NFKD", char)
        bases = [part for part in decomposed if not unicodedata.combining(part)]
        replacement = "".join(bases).lower()
        # Portuguese letters decompose to one base codepoint.  Keep a strict
        # 1:1 view even if a future input contains an expanding ligature.
        out.append(replacement if len(replacement) == 1 else char.lower())
    normalized = "".join(out)
    if len(normalized) != len(text):
        raise AssertionError("normalização alterou o comprimento do texto")
    return normalized


def read_text_exact(path: Path) -> str:
    """Decode UTF-8 without universal-newline translation.

    ``Path.read_text`` turns CRLF into LF.  Opening with ``newline=''`` keeps
    the codepoint stream used by the challenge offsets unchanged.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


@dataclass(frozen=True)
class Citation:
    start: int
    end: int
    kind: str
    classification: str
    canonical_id: str | None
    confidence: float

    def to_json(self, source: str, citation_id: str) -> dict[str, object]:
        return {
            "citacao_id": citation_id,
            "inicio": self.start,
            "fim": self.end,
            "trecho": source[self.start : self.end],
            "tipo": self.kind,
            "classificacao": self.classification,
            "resolucao": (
                {"id_canonico": self.canonical_id}
                if self.classification == "real"
                else None
            ),
            "confianca": self.confidence,
        }

    def encode(self) -> str:
        return ",".join(
            [
                str(self.start),
                str(self.end),
                self.classification,
                self.canonical_id or "-",
                f"{self.confidence:.4f}",
            ]
        )


@dataclass(frozen=True)
class RawMatch:
    start: int
    end: int
    kind: str
    explicit_type: str
    priority: int


@dataclass(frozen=True)
class IndexedHit:
    position: int
    tribunal: str
    year: int | None
    reporter: str
    families: frozenset[str]


def _compile(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE | re.MULTILINE | re.VERBOSE)


CASE_PREFIX = r"""
(?:
    embargos\s+de\s+declaracao\s+no\s+agravo\s+interno\s+no\s+agravo\s+em\s+recurso\s+especial
  | embargos\s+de\s+declaracao\s+no\s+recurso\s+em\s+mandado\s+de\s+seguranca
  | agravo\s+interno\s+na\s+suspensao\s+de\s+liminar\s+e\s+de\s+sentenca
  | agravo\s+interno\s+no\s+agravo\s+em\s+recurso\s+especial
  | agravo\s+regimental\s+no\s+agravo\s+de\s+instrumento
  | edcl\s+nos\s+edcl\s+no\s+agint\s+no\s+agravo\s+em\s+recurso\s+especial
  | edcl\s+no\s+agint\s+no\s+(?:agravo\s+em\s+recurso\s+especial|recurso\s+especial|aresp|resp)
  | agint\s+nos\s+edcl\s+no\s+(?:recurso\s+especial|rec\.?\s*esp\.?|resp)
  | agint\s+no\s+(?:recurso\s+especial|agravo\s+em\s+recurso\s+especial|aresp|resp)
  | agint\s+na\s+(?:reclamacao|rcl)
  | agrg\s+no\s+(?:rec\.?\s*esp\.?|aresp|resp|h\.?c\.?)
  | eds?\s+no\s+agr-?respe
  | recurso\s+em\s+habeas\s+corpus
  | recurso\s+especial\s+eleitoral
  | agravo\s+em\s+recurso\s+especial
  | recurso\s+especial
  | terceiro\s+ag\.?reg\.?\s+na\s+rcl
  | agr-?respe
  | agr-?ai
  | hc
  | aresp(?:e[il])?
  | a\.?\s*resp
  | agre?sp
  | agint
  | ag\.?\s*int\.?
  | rec\.?\s*esp\.?
  | r\.?esp\.?
  | re[5s]p
  | respe\.?
  | reclamacao
  | recl\.?
  | rcl
  | rhc
  | rms
  | rse
  | apl
  | ar
  | re\.?
)
"""

EXPLICIT_CASE_RE = _compile(
    rf"(?<![a-z0-9])(?:(?:ed(?:cl|s)?|agrg?|agint)\s+nos?\s+)*{CASE_PREFIX}{NUMBER_MARKER}\s+{NUMBER_BODY}{STATE_SUFFIX}"
    rf"(?![a-z0-9])"
)

TST_CASE_RE = _compile(
    rf"(?<![a-z0-9])"
    rf"(?:processo\s+n(?:[.o]\s*)?\s*)?"
    rf"(?:tst\s*-\s*)?"
    rf"(?:(?:ed|e|ag|ai)\s*-\s*)*"
    rf"(?:agarr|airr|arr|rr)\s*-\s*"
    rf"{NUMBER_BODY}"
)

TSE_RP_RE = _compile(rf"(?<![a-z0-9])r\s*-\s*rp{NUMBER_MARKER}\s+{NUMBER_BODY}")

SUMULA_RE = _compile(
    r"(?<![a-z0-9])(?:su(?:m|rn)u[l1]a|5umula|sum\.)\s+"
    r"(?:vinculante\s+)?(?:n(?:[.o]\s*)?\s*)?"
    r"[0-9olsig]+(?:\s+do\s+(?:stf|stj|tst|tse))?"
)

TEMA_RE = _compile(rf"(?<![a-z0-9])tema\s+{NUMBER_BODY}\s+da\s+repercussao\s+geral")

LAW_RE = _compile(
    r"(?<![a-z0-9])art(?:igo|\.)?\s*"
    r"[0-9]+(?:\.[0-9]{3})?o?"
    r"(?:\s*,\s*(?:§\s*[0-9]+o?(?:-[a-z])?|inciso\s+[ivxlcdm]+|"
    r"[ivxlcdm]+|['\"]?[a-z]['\"]?))*"
    r"\s*,?\s*(?:do|da)\s+"
    r"(?:"
    r"codig[0o]\s+(?:de\s+)?(?:processo\s+penal|processo\s+civil|defesa\s+do\s+consumidor|"
    r"penal\s+militar|eleitoral|civil)"
    r"|consolidacao\s+das\s+leis\s+do\s+trabalho"
    r"|constituicao(?:\s+da\s+republica|\s+fed.ral)?"
    r"|lei\s+complementar\s+n(?:[.o]\s*)?\s*[0-9](?:[0-9./-]*[0-9])?"
    r"|lei\s+(?:n(?:[.o]\s*)?\s*)?[0-9](?:[0-9./-]*[0-9])?"
    r"|cpc|clt"
    r")"
)


# Incomplete citations are built compositionally from a legal semantic head
# and a generic noun-phrase boundary.  No complete phrase from the public
# development set is stored here.
SEMANTIC_HEAD_RE = _compile(
    r"(?<![a-z0-9])(?:"
    r"orientacao\s+jurisprud[a-z]{3,9}"
    r"|ent[a-z]{4,10}ento\s+(?:sumulad[a-z]*|jurisprud[a-z]*|"
    r"consolidad[a-z]*|reiterad[a-z]*)"
    r"|(?:verbete|enunciado)\s+sumular"
    r"|jurisprud[a-z]{3,9}"
    r"|(?:recent[a-z]*|reiterad[a-z]*)\s+(?:acordao|decisao|precedentes?)"
    r"|precedentes?|acordao|decisao|julgado"
    r"|normas?|legislacao|dispositivos?|artigo|lei"
    r")(?![a-z0-9])"
)

DESCRIPTIVE_CASE_RE = _compile(
    rf"(?<![a-z0-9]){CASE_PREFIX}"
    rf"(?=\s+(?:do\s+(?:stf|stj|tst|tse|stm)\b|de\s+20[0-9]{{2}}\b))"
)

_PHRASE_ABBREVIATIONS = {
    "a",
    "ag",
    "art",
    "esp",
    "int",
    "min",
    "n",
    "no",
    "re",
    "rec",
    "recl",
    "rel",
    "rcl",
    "sum",
    "sumula",
}


def _trim_span(view: str, start: int, end: int) -> tuple[int, int]:
    while start < end and view[start].isspace():
        start += 1
    while end > start and (view[end - 1].isspace() or view[end - 1] in ",;:"):
        end -= 1
    return start, end


def _semantic_terminal_period(view: str, start: int, stop: int) -> int | None:
    """Find a sentence period while ignoring initials and legal abbreviations."""
    for match in re.finditer(r"\.", view[start:stop]):
        position = start + match.start()
        if position and position + 1 < len(view):
            if view[position - 1].isdigit() and view[position + 1].isdigit():
                continue
        tail = view[position + 1 :]
        whitespace_match = re.match(r"\s*", tail)
        whitespace = whitespace_match.group() if whitespace_match else ""
        if tail and not whitespace:
            continue
        previous = re.search(
            r"([a-z0-9]+)\s*$", view[max(start, position - 16) : position]
        )
        if previous and (
            len(previous.group(1)) == 1 or previous.group(1) in _PHRASE_ABBREVIATIONS
        ):
            continue
        return position
    return None


def _semantic_phrase_end(view: str, start: int, *, limit: int = 190) -> int | None:
    """Return the nearest generic boundary of an incomplete citation phrase."""
    stop = min(len(view), start + limit)
    candidates: list[int] = []

    for pattern in (
        r"\n\s*\n",
        r";",
        r"\s+(?:para\s+sustentar|foi\s+reafirmad[a-z]*|"
        r"afastou\s+(?:a\s+)?pretens[a-z]*|"
        r"(?:antes|depois)\s+(?:d[oa]s?|de))\b",
    ):
        boundary = re.search(pattern, view[start:stop])
        if boundary:
            candidates.append(start + boundary.start())

    # A topic adjunct after a named court/source describes the surrounding
    # argument, not the identity of an already complete vague reference.
    topic = re.search(
        r"\b(?:tribunal|corte|stf|stj|tst|tse|stm)\b[^,.;\n]{0,50}?"
        r"(?P<boundary>\s+sobre\s+(?:a|o)\s+(?:materia|questao|tema|controversia))\b",
        view[start:stop],
    )
    if topic:
        candidates.append(start + topic.start("boundary"))

    period = _semantic_terminal_period(view, start, stop)
    if period is not None:
        candidates.append(period)

    for comma in re.finditer(",", view[start:stop]):
        position = start + comma.start()
        tail = view[position + 1 : stop]
        if re.match(
            r"\s*(?:de\s+20[0-9]{2}\b|(?:da|pela|sob)\s+relatoria\b|"
            r"rel\.?\s*(?:min\.?\s*)?)",
            tail,
        ):
            continue
        candidates.append(position)
        break

    if not candidates:
        return None
    _unused_start, end = _trim_span(view, start, min(candidates))
    while end > start and view[end - 1] == ".":
        end -= 1
    return end


def _semantic_incomplete_kind(view: str, start: int, end: int) -> str | None:
    """Classify a bounded phrase as an incomplete law/case reference."""
    phrase = view[start:end]
    if not phrase or len(phrase) > 190:
        return None

    # Final criteria exclude generic appeals to legislation/jurisprudence.
    # A descriptive case citation must identify a source via procedural
    # class/court plus a year and named reporter, even without a case number.
    legal_head = DESCRIPTIVE_CASE_RE.match(phrase) or re.match(
        r"(?:recent[a-z]*\s+)?(?:precedente|acordao|decisao|julgado)\s+do\s+"
        r"(?:stf|stj|tst|tse|stm)\b", phrase
    )
    year = re.search(r"\b(?:19|20)[0-9]{2}\b", phrase)
    reporter = re.search(
        r"\b(?:relatoria\s+d[aeo]|rel\.?\s*(?:min\.?\s*)?)\s*[a-z]{2,}",
        phrase,
    )
    return "jurisprudencia" if legal_head and year and reporter else None


def _overlap(a: RawMatch, b: RawMatch) -> int:
    return max(0, min(a.end, b.end) - max(a.start, b.start))


def _is_contextual_distractor(view: str, item: RawMatch) -> bool:
    """Reject identifiers explicitly presented as administrative/non-legal codes."""
    if item.explicit_type != "case":
        return False
    prefix = view[max(0, item.start - 80) : item.start]
    return bool(
        re.search(
            r"(?:\bcodigo|\barquivo(?:\s+interno)?(?:\s+denominado)?|"
            r"\bprotocolo|\bsenha|\bprontuario)\s*[:#-]?\s*$",
            prefix,
        )
    )


def _raw_match_for_span(view: str, start: int, end: int, priority: int) -> RawMatch:
    literal = view[start:end]
    if LAW_RE.fullmatch(literal):
        return RawMatch(start, end, "lei", "law", priority)
    if SUMULA_RE.fullmatch(literal) or TEMA_RE.fullmatch(literal):
        return RawMatch(start, end, "jurisprudencia", "sumula", priority)
    if any(
        regex.fullmatch(literal) for regex in (TST_CASE_RE, TSE_RP_RE, EXPLICIT_CASE_RE)
    ):
        return RawMatch(start, end, "jurisprudencia", "case", priority)
    law_cues = (
        "norma",
        "legislacao",
        "dispositivo",
        "artigo correspondente",
        "lei que",
    )
    kind = "lei" if literal.startswith(law_cues) else "jurisprudencia"
    return RawMatch(start, end, kind, "incomplete", priority)


def extract_candidates(
    source: str, *, use_dev_templates: bool = False
) -> list[RawMatch]:
    view = matching_view(source)
    matches: list[RawMatch] = []

    def collect(
        regex: re.Pattern[str], kind: str, explicit_type: str, priority: int
    ) -> None:
        for match in regex.finditer(view):
            start, end = _trim_span(view, match.start(), match.end())
            if end > start:
                matches.append(RawMatch(start, end, kind, explicit_type, priority))

    collect(LAW_RE, "lei", "law", 80)
    collect(SUMULA_RE, "jurisprudencia", "sumula", 80)
    collect(TEMA_RE, "jurisprudencia", "sumula", 80)
    collect(TST_CASE_RE, "jurisprudencia", "case", 70)
    collect(TSE_RP_RE, "jurisprudencia", "case", 70)
    collect(EXPLICIT_CASE_RE, "jurisprudencia", "case", 60)

    semantic_starts = {
        (match.start(), match.end()) for match in SEMANTIC_HEAD_RE.finditer(view)
    }
    semantic_starts.update(
        (match.start(), match.end()) for match in DESCRIPTIVE_CASE_RE.finditer(view)
    )
    for start, _head_end in sorted(semantic_starts):
        end = _semantic_phrase_end(view, start)
        if end is None:
            continue
        kind = _semantic_incomplete_kind(view, start, end)
        if kind:
            matches.append(RawMatch(start, end, kind, "incomplete", 70))

    # Kept only as an explicit development diagnostic.  These prose templates
    # were discovered in the public development corpus and therefore are not a
    # defensible assumption for unseen documents.
    if use_dev_templates:
        from template_extractor import extract_spans as extract_template_spans

        matches.extend(
            _raw_match_for_span(view, start, end, priority=110)
            for start, end in extract_template_spans(source)
        )

    # Exclude the structured preamble (own-case number, OAB, protocol, pages),
    # but do not assume an absolute character offset for future documents.
    first_body_codepoint = body_start(source)
    matches = [
        item
        for item in matches
        if item.start >= first_body_codepoint
        and not _is_contextual_distractor(view, item)
    ]

    # Resolve overlaps deterministically: semantic priority, then longer span.
    selected: list[RawMatch] = []
    for item in sorted(
        matches,
        key=lambda value: (
            -(value.explicit_type != "incomplete"),
            -value.priority,
            -(value.end - value.start),
            value.start,
        ),
    ):
        if any(_overlap(item, old) > 0 for old in selected):
            continue
        selected.append(item)
    return sorted(selected, key=lambda value: (value.start, value.end))


NUMBER_TOKEN_RE = re.compile(r"(?<!\w)\d(?:[\d\s./\-–—º°]*\d)?")


def _reference_families(view: str, tribunal: str = "") -> frozenset[str]:
    """Infer coarse procedural families from a citation or header context."""
    families: set[str] = set()
    if tribunal.upper() == "TST" or re.search(
        r"(?:\btst\s*-|(?<![a-z])(?:airr|arr|rr)\s*-)", view
    ):
        families.add("tst")
    if re.search(r"recurso\s+especial\s+eleitoral|respe|arespe[il]", view):
        families.add("respe")
    elif re.search(r"agravo\s+em\s+recurso\s+especial|aresp|a\.?\s*resp", view):
        families.add("aresp")
    elif re.search(r"recurso\s+especial|rec\.?\s*esp\.?|r\.?esp\.?|resp", view):
        families.add("resp")
    if re.search(r"recurso\s+em\s+habeas\s+corpus|\brhc\b", view):
        families.add("rhc")
    if re.search(r"recurso\s+em\s+mandado\s+de\s+seguranca|\brms\b", view):
        families.add("rms")
    if re.search(r"reclamacao|\brcl\b|\brecl\.", view):
        families.add("rcl")
    if re.search(r"recurso\s+em\s+sentido\s+estrito|\brse\b", view):
        families.add("rse")
    if re.search(r"apelacao|\bapl\b", view):
        families.add("apl")
    if re.search(
        r"agravo\s+(?:regimental\s+no\s+)?agravo\s+de\s+instrumento|agr-?ai", view
    ):
        families.add("ai")
    if re.search(r"agravo\s+interno|\bagint\b|ag\.?\s*int\.?", view):
        families.add("agint")
    if re.search(r"acao\s+rescisoria|(?<![a-z])ar(?=\s|n?\.?\s*[0-9])", view):
        families.add("ar")
    if re.search(r"suspensao\s+de\s+liminar\s+e\s+de\s+sentenca", view):
        families.add("sls")
    if re.search(r"recurso\s+na\s+representacao|r\s*-\s*rp", view):
        families.add("rp")
    if re.search(r"habeas\s+corpus|(?<![a-z])h\.?c\.?(?=\s|n)", view):
        families.add("hc")
    if re.search(r"recurso\s+extraordinario|(?<![a-z])re\.?(?=\s+n|\s+[0-9])", view):
        families.add("re")
    return frozenset(families)


class CanonicalResolver:
    """Resolve explicit references against the frozen challenge database."""

    _INDEX_CACHE: ClassVar[
        dict[
            Path,
            tuple[
                dict[str, dict[str, IndexedHit]],
                CanonicalMetadataIndex,
            ],
        ]
    ] = {}

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path.resolve()
        cached = self._INDEX_CACHE.get(self.database_path)
        if cached is not None:
            self.number_index, self.metadata_index = cached
            return

        number_index: dict[str, dict[str, IndexedHit]] = defaultdict(dict)
        database_uri = self.database_path.as_uri() + "?mode=ro"
        connection = sqlite3.connect(database_uri, uri=True)
        try:
            rows = connection.execute(
                "SELECT id, tribunal, ano, relator, texto "
                "FROM documentos WHERE natureza = 'acordao'"
            )
            for canonical_id, tribunal, year, reporter, text in rows:
                id_text = str(canonical_id)
                max_position = 5000 if tribunal == "TST" else 400
                for match in NUMBER_TOKEN_RE.finditer(text):
                    if match.start() > max_position:
                        break
                    key = "".join(char for char in match.group() if char.isdigit())
                    if len(key) < 3:
                        continue
                    context = matching_view(
                        text[max(0, match.start() - 140) : match.end() + 80]
                    )
                    families = _reference_families(context, tribunal or "")
                    if not families:
                        continue
                    hit = IndexedHit(
                        position=match.start(),
                        tribunal=tribunal or "",
                        year=int(year) if year is not None else None,
                        reporter=str(reporter or ""),
                        families=families,
                    )
                    old = number_index[key].get(id_text)
                    if old is None or hit.position < old.position:
                        number_index[key][id_text] = hit
        finally:
            connection.close()
        self.number_index = number_index
        self.metadata_index = CanonicalMetadataIndex(self.database_path)
        self._INDEX_CACHE[self.database_path] = (
            self.number_index,
            self.metadata_index,
        )

    def resolve_law(self, literal: str) -> str | None:
        return self.metadata_index.resolve_law(literal)

    def resolve_sumula(self, literal: str) -> str | None:
        return self.metadata_index.resolve_summary(literal)

    @staticmethod
    def identifier_variants(literal: str) -> list[str]:
        view = matching_view(literal)
        view = re.sub(
            rf"(?:[/\-()]|\s)+(?:{COURTS})\)?\s*$",
            "",
            view,
            flags=re.IGNORECASE,
        )
        chunks = re.findall(r"(?<!\w)\d[\d\s./\-olsg]*", view)
        variants: set[str] = set()
        mapping: dict[str, tuple[str, ...]] = {
            "o": ("0",),
            "l": ("1",),
            "i": ("1",),
            "s": ("5",),
            "g": ("9", "6"),
        }
        for chunk in chunks:
            options: list[tuple[str, ...]] = []
            combination_count = 1
            for char in chunk:
                if char.isdigit():
                    options.append((char,))
                elif char in mapping:
                    options.append(mapping[char])
                    combination_count *= len(mapping[char])
            if not options or len(options) > 30 or combination_count > 64:
                continue
            for product in itertools.product(*options):
                key = "".join(product)
                if len(key) >= 3:
                    variants.add(key)
        return sorted(variants, key=lambda value: (-len(value), value))

    def resolve_case(self, literal: str) -> tuple[str, str | None]:
        variants = self.identifier_variants(literal)
        if not variants:
            return "incompleta", None
        query_families = _reference_families(matching_view(literal))
        candidates: dict[str, IndexedHit] = {}
        for key in variants:
            for canonical_id, hit in self.number_index.get(key, {}).items():
                if query_families and not query_families.intersection(hit.families):
                    continue
                old = candidates.get(canonical_id)
                if old is None or hit.position < old.position:
                    candidates[canonical_id] = hit
        if not candidates:
            return "inventada", None

        best_position = min(hit.position for hit in candidates.values())
        near_best = [
            canonical_id
            for canonical_id, hit in candidates.items()
            if hit.position <= best_position + 10
        ]
        if len(near_best) > 1:
            metadata = {
                (
                    candidates[canonical_id].tribunal,
                    candidates[canonical_id].year,
                    candidates[canonical_id].reporter.casefold(),
                )
                for canonical_id in near_best
            }
            if len(metadata) > 1:
                return "incompleta", None
        # Near-identical source duplicates can shift the same header by one
        # codepoint.  The lowest canonical id is the stable canonical member.
        return "real", min(near_best, key=int)


def classify(raw: RawMatch, source: str, resolver: CanonicalResolver) -> Citation:
    literal = source[raw.start : raw.end]
    if raw.explicit_type == "incomplete":
        return Citation(raw.start, raw.end, raw.kind, "incompleta", None, 1.0)
    if raw.explicit_type == "law":
        canonical_id = resolver.resolve_law(literal)
    elif raw.explicit_type == "sumula":
        canonical_id = resolver.resolve_sumula(literal)
    else:
        classification, canonical_id = resolver.resolve_case(literal)
        return Citation(
            raw.start,
            raw.end,
            raw.kind,
            classification,
            canonical_id,
            1.0,
        )
    classification = "real" if canonical_id else "inventada"
    return Citation(raw.start, raw.end, raw.kind, classification, canonical_id, 1.0)


def predict_document(
    source: str,
    resolver: CanonicalResolver,
    *,
    use_dev_templates: bool = False,
) -> list[Citation]:
    return [
        classify(raw, source, resolver)
        for raw in extract_candidates(source, use_dev_templates=use_dev_templates)
    ]


def _validate_predictions(
    citations: Iterable[Citation], source: str, document_id: str
) -> None:
    ordered = sorted(citations, key=lambda item: (item.start, item.end))
    for index, citation in enumerate(ordered):
        if not (0 <= citation.start < citation.end <= len(source)):
            raise ValueError(
                f"{document_id}: span inválido {citation.start}:{citation.end}"
            )
        if citation.classification == "real" and not (
            citation.canonical_id and citation.canonical_id.isdigit()
        ):
            raise ValueError(f"{document_id}: citação real sem id canônico válido")
        for other in ordered[index + 1 :]:
            intersection = max(
                0, min(citation.end, other.end) - max(citation.start, other.start)
            )
            if not intersection:
                break
            union = (
                (citation.end - citation.start)
                + (other.end - other.start)
                - intersection
            )
            if intersection / union >= 0.5:
                raise ValueError(f"{document_id}: predições duplicadas/sobrepostas")


def run_pipeline(
    text_dir: Path,
    database_path: Path,
    json_dir: Path,
    submission_path: Path,
    *,
    use_dev_templates: bool = False,
) -> None:
    inputs = sorted(text_dir.glob("*.txt"))
    if not inputs:
        raise ValueError(f"nenhum .txt encontrado em {text_dir}")
    resolver = CanonicalResolver(database_path)
    json_dir.mkdir(parents=True, exist_ok=True)
    expected_stems = {path.stem for path in inputs}
    stale_json = sorted(
        path.name for path in json_dir.glob("*.json") if path.stem not in expected_stems
    )
    if stale_json:
        raise ValueError(
            f"{json_dir} contém JSONs de outra execução: {stale_json}; "
            "use uma pasta de saída vazia para não misturar conjuntos"
        )
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_rows: list[tuple[str, str]] = []

    for input_path in inputs:
        document_id = input_path.stem
        source = read_text_exact(input_path)
        citations = predict_document(
            source,
            resolver,
            use_dev_templates=use_dev_templates,
        )
        _validate_predictions(citations, source, document_id)
        payload = {
            "schema_version": "1.2",
            "documento_id": document_id,
            "citacoes": [
                citation.to_json(source, f"p{index}")
                for index, citation in enumerate(citations, start=1)
            ],
        }
        (json_dir / f"{document_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        encoded = "|".join(citation.encode() for citation in citations) or "-"
        submission_rows.append((document_id, encoded))

    with submission_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["documento_id", "citacoes"])
        writer.writerows(submission_rows)
    print(f"JSONs: {json_dir} ({len(submission_rows)} documentos)")
    print(f"Submission: {submission_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texts", type=Path, default=ROOT / "txt")
    parser.add_argument("--database", type=Path, default=ROOT / "desafio1_bracis.db")
    parser.add_argument(
        "--json-dir", type=Path, default=ROOT / "artifacts" / "blind_json"
    )
    parser.add_argument(
        "--submission",
        type=Path,
        default=ROOT / "artifacts" / "submission_blind.csv",
    )
    parser.add_argument(
        "--use-dev-templates",
        action="store_true",
        help=(
            "ativa templates de prosa derivados do conjunto público apenas "
            "para diagnóstico; desativado por padrão"
        ),
    )
    args = parser.parse_args()
    run_pipeline(
        args.texts,
        args.database,
        args.json_dir,
        args.submission,
        use_dev_templates=args.use_dev_templates,
    )


if __name__ == "__main__":
    main()
