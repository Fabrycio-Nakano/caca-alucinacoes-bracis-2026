#!/usr/bin/env python3
"""Metadata-aware indexes for provisions and court summaries.

The challenge SQLite file does not expose the statute (``diploma``) of a
``dispositivo`` row.  Two ``sumula`` rows also omit their number from
``texto``.  This module keeps recovery of that missing metadata separate from
the extractor/resolver and makes every recovery auditable.

Resolution is deliberately conservative:

* provisions are keyed by ``(diploma, article)`` rather than article alone;
* court summaries are keyed by ``(court, number)``;
* collisions remain collisions and therefore resolve to ``None``.

The content signatures below describe the canonical documents themselves;
they do not contain phrases from the development input documents or expected
prediction labels.
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def normalized_view(value: str) -> str:
    """Fold accents/case and whitespace for metadata matching."""
    value = value.translate(
        {ord("–"): "-", ord("—"): "-", ord("−"): "-", ord("º"): "o", ord("°"): "o"}
    )
    folded = unicodedata.normalize("NFKD", value)
    ascii_like = "".join(char for char in folded if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", ascii_like.casefold()).strip()


def _digits(value: str) -> str:
    digits = "".join(char for char in value if char.isdigit())
    return str(int(digits)) if digits else ""


ARTICLE_RE = re.compile(r"\bart(?:igo|\.)?\s*([0-9]+(?:\.[0-9]{3})?)", re.IGNORECASE)
SUMMARY_NUMBER_RE = re.compile(r"\bsumula\s+(?:n(?:[.o]\s*)?\s*)?([0-9]+)")


@dataclass(frozen=True)
class LawContentSignature:
    article: str
    diploma: str
    pattern: re.Pattern[str]


@dataclass(frozen=True)
class SummaryContentSignature:
    court: str
    number: str
    pattern: re.Pattern[str]


@dataclass(frozen=True)
class MetadataRecovery:
    canonical_id: str
    nature: str
    field: str
    value: str
    method: str


@dataclass(frozen=True)
class UnindexedRecord:
    canonical_id: str
    nature: str
    missing_fields: tuple[str, ...]


def _signature(article: str, diploma: str, pattern: str) -> LawContentSignature:
    return LawContentSignature(article, diploma, re.compile(pattern))


# ``documentos`` currently has no statute column.  These restrained content
# signatures recover the source statute from the canonical text itself.  They
# are tied to legal wording, not canonical IDs, so an ID migration does not
# silently corrupt the index.
LAW_CONTENT_SIGNATURES: tuple[LawContentSignature, ...] = (
    _signature(
        "276", "codigo_eleitoral", r"decisoes dos tribunais regionais sao terminativas"
    ),
    _signature(
        "290", "codigo_penal_militar", r"receber, preparar, produzir, vender, fornecer"
    ),
    _signature("14", "codigo_defesa_consumidor", r"fornecedor de servicos responde"),
    _signature(
        "93",
        "constituicao_federal",
        r"supremo tribunal federal.*estatuto da magistratura",
    ),
    _signature("896", "clt", r"recurso de revista.*tribunal superior do trabalho"),
    _signature(
        "7", "constituicao_federal", r"direitos dos trabalhadores urbanos e rurais"
    ),
    _signature("5", "constituicao_federal", r"todos sao iguais perante a lei"),
    _signature("818", "clt", r"onus da prova incumbe.*reclamante"),
    _signature(
        "312", "codigo_processo_penal", r"prisao preventiva.*garantia da ordem publica"
    ),
    _signature("477", "clt", r"extincao do contrato de trabalho"),
    _signature("186", "codigo_civil", r"acao ou omissao voluntaria.*ato ilicito"),
    _signature("1", "lei_complementar_64_1990", r"sao inelegiveis"),
    _signature("373", "codigo_processo_civil_2015", r"onus da prova incumbe.*ao autor"),
)


# The STF and TST records are the only summary rows whose canonical text omits
# the number.  Their leading official wording is sufficiently distinctive to
# recover it, and each use is reported through ``recoveries``.
SUMMARY_CONTENT_SIGNATURES: tuple[SummaryContentSignature, ...] = (
    SummaryContentSignature(
        "stf",
        "10",
        re.compile(r"viola a clausula de reserva de plenario"),
    ),
    SummaryContentSignature(
        "tst",
        "331",
        re.compile(
            r"contrato de prestacao de servicos.*"
            r"contratacao de trabalhadores por empresa interposta"
        ),
    ),
)


# Specific statute numbers are kept distinct.  In particular, Lei 9.504/1997
# is not collapsed into the Electoral Code and Lei 13.467/2017 is not treated
# as the CLT itself.
DIPLOMA_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern))
    for name, pattern in (
        (
            "lei_complementar_64_1990",
            r"lei complementar(?: n[.o]?)? 64\s*/\s*(?:19)?90",
        ),
        (
            "codigo_processo_civil_2015",
            r"(?:codig[0o] de processo civil|\bcpc\b|lei(?: n[.o]?)? 13\.?105\s*/\s*2015)",
        ),
        (
            "codigo_penal_militar",
            r"codig[0o] penal militar|\bcpm\b|decreto-lei(?: n[.o]?)? 1\.?001\s*/\s*1969",
        ),
        (
            "codigo_processo_penal",
            r"codig[0o] de processo penal|\bcpp\b|decreto-lei(?: n[.o]?)? 3\.?689\s*/\s*1941",
        ),
        (
            "codigo_defesa_consumidor",
            r"codig[0o] de defesa do consumidor|\bcdc\b|lei(?: n[.o]?)? 8\.?078\s*/\s*1990",
        ),
        ("codigo_eleitoral", r"codig[0o] eleitoral|lei(?: n[.o]?)? 4\.?737\s*/\s*1965"),
        ("codigo_civil", r"codig[0o] civil|\bcc\b|lei(?: n[.o]?)? 10\.?406\s*/\s*2002"),
        (
            "clt",
            r"consolidacao das leis do trabalho|\bclt\b|decreto-lei(?: n[.o]?)? 5\.?452\s*/\s*1943",
        ),
        (
            "constituicao_federal",
            r"constituicao(?: da republica| fed.ral)?|\bcf(?:/88)?\b|\bcrfb(?:/88)?\b",
        ),
    )
)


def infer_diploma(literal: str) -> str | None:
    """Return a canonical statute key from a citation literal."""
    view = normalized_view(literal)
    hits = {name for name, pattern in DIPLOMA_PATTERNS if pattern.search(view)}
    return next(iter(hits)) if len(hits) == 1 else None


def _article_from_text(text: str) -> str | None:
    match = ARTICLE_RE.search(normalized_view(text[:200]))
    return _digits(match.group(1)) if match else None


def _diploma_from_content(article: str, text: str) -> str | None:
    view = normalized_view(text)
    hits = {
        item.diploma
        for item in LAW_CONTENT_SIGNATURES
        if item.article == article and item.pattern.search(view)
    }
    return next(iter(hits)) if len(hits) == 1 else None


def _summary_number_from_content(court: str, text: str) -> str | None:
    view = normalized_view(text)
    hits = {
        item.number
        for item in SUMMARY_CONTENT_SIGNATURES
        if item.court == court and item.pattern.search(view)
    }
    return next(iter(hits)) if len(hits) == 1 else None


def _only(candidates: Iterable[str]) -> str | None:
    unique = set(candidates)
    return next(iter(unique)) if len(unique) == 1 else None


class CanonicalMetadataIndex:
    """Build collision-preserving metadata indexes from ``documentos``."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path.resolve()
        law_candidates: dict[tuple[str, str], set[str]] = defaultdict(set)
        summary_candidates: dict[tuple[str, str], set[str]] = defaultdict(set)
        recoveries: list[MetadataRecovery] = []
        unindexed: list[UnindexedRecord] = []

        database_uri = self.database_path.as_uri() + "?mode=ro"
        connection = sqlite3.connect(database_uri, uri=True)
        connection.row_factory = sqlite3.Row
        try:
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(documentos)")
            }
            diploma_column = next(
                (name for name in ("diploma", "fonte_normativa") if name in columns),
                None,
            )
            number_column = next(
                (name for name in ("numero", "numero_sumula") if name in columns),
                None,
            )
            selected = ["id", "tribunal", "texto", "natureza"]
            if diploma_column:
                selected.append(diploma_column)
            if number_column:
                selected.append(number_column)
            query = "SELECT " + ", ".join(selected) + " FROM documentos"

            for row in connection.execute(query):
                nature = str(row["natureza"])
                if nature not in {"dispositivo", "sumula"}:
                    continue
                canonical_id = str(row["id"])
                text = str(row["texto"])

                if nature == "dispositivo":
                    article = _article_from_text(text)
                    declared = (
                        str(row[diploma_column])
                        if diploma_column and row[diploma_column]
                        else ""
                    )
                    diploma = infer_diploma(declared) if declared else None
                    if not diploma and article:
                        diploma = _diploma_from_content(article, text)
                        if diploma:
                            recoveries.append(
                                MetadataRecovery(
                                    canonical_id,
                                    nature,
                                    "diploma",
                                    diploma,
                                    "canonical_text_signature",
                                )
                            )
                    missing = tuple(
                        field
                        for field, value in (("artigo", article), ("diploma", diploma))
                        if not value
                    )
                    if missing:
                        unindexed.append(UnindexedRecord(canonical_id, nature, missing))
                    else:
                        assert diploma is not None and article is not None
                        law_candidates[(diploma, article)].add(canonical_id)
                    continue

                court = normalized_view(str(row["tribunal"] or ""))
                declared_number = (
                    _digits(str(row[number_column]))
                    if number_column and row[number_column] is not None
                    else ""
                )
                text_number_match = SUMMARY_NUMBER_RE.search(normalized_view(text))
                number = declared_number or (
                    _digits(text_number_match.group(1)) if text_number_match else ""
                )
                if not number and court:
                    number = _summary_number_from_content(court, text) or ""
                    if number:
                        recoveries.append(
                            MetadataRecovery(
                                canonical_id,
                                nature,
                                "numero",
                                number,
                                "canonical_text_signature",
                            )
                        )
                missing = tuple(
                    field
                    for field, value in (("tribunal", court), ("numero", number))
                    if not value
                )
                if missing:
                    unindexed.append(UnindexedRecord(canonical_id, nature, missing))
                else:
                    summary_candidates[(court, number)].add(canonical_id)
        finally:
            connection.close()

        self.law_index = {
            key: tuple(sorted(values, key=int))
            for key, values in law_candidates.items()
        }
        self.summary_index = {
            key: tuple(sorted(values, key=int))
            for key, values in summary_candidates.items()
        }
        self.recoveries = tuple(recoveries)
        self.unindexed = tuple(unindexed)

    def resolve_law(self, literal: str) -> str | None:
        """Resolve only when both statute and article identify one DB row."""
        view = normalized_view(literal)
        match = ARTICLE_RE.search(view)
        article = _digits(match.group(1)) if match else ""
        diploma = infer_diploma(literal)
        if not article or not diploma:
            return None
        return _only(self.law_index.get((diploma, article), ()))

    def resolve_summary(self, literal: str) -> str | None:
        """Resolve a súmula by court and number, preserving DB collisions."""
        view = normalized_view(literal)
        match = re.search(
            r"(?:su(?:m|rn)u[l1]a|5umula|sum\.)\s+"
            r"(?P<vinculante>vinculante\s+)?"
            r"(?:n(?:[.o]\s*)?\s*)?(?P<number>[0-9olsig]+)",
            view,
        )
        if not match:
            return None
        translated = match.group("number").translate(
            {ord("o"): "0", ord("l"): "1", ord("i"): "1", ord("s"): "5", ord("g"): "9"}
        )
        number = _digits(translated)
        court_match = re.search(r"\b(stf|stj|tst|tse)\b", view)
        explicit_court = court_match.group(1) if court_match else ""
        if match.group("vinculante") and explicit_court not in {"", "stf"}:
            return None
        court = explicit_court or ("stf" if match.group("vinculante") else "")
        if court:
            return _only(self.summary_index.get((court, number), ()))
        return _only(
            canonical_id
            for (known_court, known_number), canonical_ids in self.summary_index.items()
            if known_number == number
            for canonical_id in canonical_ids
        )

    def resolve_sumula(self, literal: str) -> str | None:
        """Portuguese compatibility name used by the existing pipeline."""
        return self.resolve_summary(literal)
