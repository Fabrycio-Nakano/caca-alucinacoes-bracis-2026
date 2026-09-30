"""Regression tests for the blind and development-oracle pipelines."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blind_pipeline import (  # noqa: E402
    CanonicalResolver,
    matching_view,
    extract_candidates,
    predict_document,
    read_text_exact,
    run_pipeline,
)
from build_dev_oracle import build_oracle  # noqa: E402
from evaluate_submission import load_solution, load_submission  # noqa: E402
from kaggle_metric import avaliar  # noqa: E402


TEXT_DIR = PROJECT_ROOT / "txt"
DATABASE_PATH = PROJECT_ROOT / "desafio1_bracis.db"
GOLD_PATH = PROJECT_ROOT / "goldenset_offsets.csv"


def _gold_rows() -> list[dict[str, str]]:
    with GOLD_PATH.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _gold_spans_by_document() -> dict[str, list[tuple[int, int]]]:
    spans: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in _gold_rows():
        spans[row["documento_id"]].append((int(row["inicio"]), int(row["fim"])))
    return {
        document_id: sorted(document_spans)
        for document_id, document_spans in spans.items()
    }


def _official_result(submission_path: Path) -> dict[str, Any]:
    solution = load_solution(GOLD_PATH)
    submission = load_submission(submission_path)
    return avaliar(solution, submission, row_id="documento_id")


def _assert_perfect_score(result: dict[str, Any]) -> None:
    assert float(result["score_final"]) == pytest.approx(1.1, rel=0.0, abs=1e-12)
    levels = result["niveis"]
    assert isinstance(levels, dict)
    assert set(levels) == {1, 2}
    for level in levels.values():
        assert level["macro_f1"] == pytest.approx(1.0)
        assert level["score"] == pytest.approx(1.1)


def test_matching_view_preserves_all_development_offsets() -> None:
    """Normalization must retain a one-to-one mapping to source codepoints."""

    rows_by_document: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _gold_rows():
        rows_by_document[row["documento_id"]].append(row)

    saw_non_breaking_space = False
    saw_multiline_citation = False
    for document_id, rows in rows_by_document.items():
        source = (TEXT_DIR / f"{document_id}.txt").read_text(encoding="utf-8")
        view = matching_view(source)

        assert len(view) == len(source)
        for row in rows:
            start, end = int(row["inicio"]), int(row["fim"])
            literal = source[start:end]
            saw_non_breaking_space |= "\u00a0" in literal
            saw_multiline_citation |= "\n" in literal
            assert view[start:end] == matching_view(literal)

    # Exercise the two offset-sensitive level-2 phenomena explicitly.
    assert saw_non_breaking_space
    assert saw_multiline_citation


def test_generic_extractor_exactly_covers_final_gold() -> None:
    """Generic rules must find every final citation without extra spans."""

    expected_by_document = _gold_spans_by_document()
    assert len(expected_by_document) == 26
    assert sum(map(len, expected_by_document.values())) == 192

    for document_id, expected in expected_by_document.items():
        source = (TEXT_DIR / f"{document_id}.txt").read_text(encoding="utf-8")
        assert [(x.start, x.end) for x in extract_candidates(source)] == expected


def test_blind_pipeline_scores_1_1_on_final_data(
    tmp_path: Path,
) -> None:
    json_dir = tmp_path / "blind_json"
    submission_path = tmp_path / "submission_blind.csv"

    run_pipeline(TEXT_DIR, DATABASE_PATH, json_dir, submission_path)

    assert submission_path.is_file()
    assert len(list(json_dir.glob("*.json"))) == 26
    result = _official_result(submission_path)
    _assert_perfect_score(result)


def test_corrected_tst_link_uses_database_without_override() -> None:
    resolver = CanonicalResolver(DATABASE_PATH)
    citation = predict_document("TST-AgARR-25823-78.2015.5.24.0091", resolver)[0]
    assert citation.classification == "real"
    assert citation.canonical_id == "1974934139"


def test_oracle_scores_1_1_and_writes_literal_json_spans(
    tmp_path: Path,
) -> None:
    json_dir = tmp_path / "oracle_json"
    submission_path = tmp_path / "submission_oracle.csv"

    build_oracle(GOLD_PATH, TEXT_DIR, json_dir, submission_path)

    _assert_perfect_score(_official_result(submission_path))

    json_paths = sorted(json_dir.glob("*.json"))
    assert len(json_paths) == 26
    citation_count = 0
    saw_literal_newline = False
    for json_path in json_paths:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        document_id = payload["documento_id"]
        source = (TEXT_DIR / f"{document_id}.txt").read_text(encoding="utf-8")

        assert payload["schema_version"] == "1.2"
        for citation in payload["citacoes"]:
            start, end = citation["inicio"], citation["fim"]
            literal = source[start:end]
            assert citation["trecho"] == literal
            saw_literal_newline |= "\n" in citation["trecho"]
            citation_count += 1

    assert citation_count == 192
    assert saw_literal_newline


def test_supported_variants_and_template_overlap() -> None:
    resolver = CanonicalResolver(DATABASE_PATH)
    cases = [
        (
            "artigo 373, inciso I, do Código de Processo Civil",
            ("lei", "real", "28893055"),
        ),
        ("Súrnula 211 do STJ", ("jurisprudencia", "real", "1289710776")),
        ("HC nº 123.456/SP", ("jurisprudencia", "inventada", None)),
        ("art. 373 da Lei 13.105/2015", ("lei", "real", "28893055")),
        ("RE 2024", ("jurisprudencia", "inventada", None)),
    ]
    for literal, expected in cases:
        source = f"X\n\n\nInvoca-se, ainda, o {literal}, no ponto em que decide."
        predictions = predict_document(source, resolver)
        assert len(predictions) == 1
        prediction = predictions[0]
        assert source[prediction.start : prediction.end] == literal
        assert (
            prediction.kind,
            prediction.classification,
            prediction.canonical_id,
        ) == expected

    # A template may include parenthetical prose after a citation.  The direct
    # explicit span must win instead of becoming an incomplete larger span.
    source = (
        "X\n\n\nInvoca-se, ainda, o HC nº 123.456/SP (julgado em 2020), "
        "no ponto em que decide."
    )
    prediction = predict_document(source, resolver)[0]
    assert source[prediction.start : prediction.end] == "HC nº 123.456/SP"


def test_exact_reader_preserves_crlf(tmp_path: Path) -> None:
    path = tmp_path / "crlf.txt"
    path.write_bytes(b"linha 1\r\nlinha 2\r\n")
    assert read_text_exact(path) == "linha 1\r\nlinha 2\r\n"
