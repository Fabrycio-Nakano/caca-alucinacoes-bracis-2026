from __future__ import annotations

import sys
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from canonical_metadata import CanonicalMetadataIndex  # noqa: E402


DATABASE_PATH = PROJECT_ROOT / "desafio1_bracis.db"


def test_missing_summary_numbers_are_recovered_transparently() -> None:
    index = CanonicalMetadataIndex(DATABASE_PATH)

    assert index.resolve_summary("Súmula Vinculante 10") == "1289712966"
    assert index.resolve_summary("Súmula 331 do TST") == "1431369957"
    assert index.resolve_summary("Súrnula 211 do STJ") == "1289710776"
    assert index.resolve_sumula("Súmula 83 do STJ") == "1289710642"

    recovered_numbers = {
        (item.canonical_id, item.field, item.value, item.method)
        for item in index.recoveries
        if item.nature == "sumula"
    }
    assert recovered_numbers == {
        ("1289712966", "numero", "10", "canonical_text_signature"),
    }


def test_summary_resolution_keeps_court_in_key() -> None:
    index = CanonicalMetadataIndex(DATABASE_PATH)

    assert index.resolve_summary("Súmula 10 do STJ") is None
    assert index.resolve_summary("Súmula Vinculante 331 do TST") is None
    assert index.resolve_summary("Súmula 979 do STF") is None


def test_law_resolution_requires_matching_statute_and_article() -> None:
    index = CanonicalMetadataIndex(DATABASE_PATH)
    real_cases = {
        "art. 373, I, do CPC": "28893055",
        "art. 290 do Código Penal Militar": "10590194",
        "art. 93, IX, da Constituição da República": "10626510",
        "art. 477 da Consolidação das Leis do Trabalho": "10710324",
        "art. 818 da CLT": "10647746",
        "art. 5º, LV, da Constituição Federal": "10641516",
        "art. 896, § 1º-A, da CLT": "10637358",
        "art. 1º, I, 'g', da Lei Complementar nº 64/1990": "11304039",
        "artigo 7º, XXIX, da Constituição Fedcral": "10641213",
        "art 276 do Código Eleitoral": "10577194",
        "artigo 186 do Código Civil": "10718759",
        "art. 14 do Código de Defesa do Consumidor": "10606184",
        "art 312 do Código de Processo Penal": "10652044",
    }
    assert {literal: index.resolve_law(literal) for literal in real_cases} == real_cases

    # Same article, wrong or absent diploma must not inherit a unique match by
    # accident.  This protects future DBs containing repeated article numbers.
    assert index.resolve_law("artigo 5 do Código Civil") is None
    assert index.resolve_law("artigo 373") is None
    assert index.resolve_law("artigo 276 da Lei 9.504/1997") is None
    assert index.resolve_law("artigo 818 da Lei 13.467/2017") is None


def test_all_current_provisions_are_indexed_without_id_overrides() -> None:
    index = CanonicalMetadataIndex(DATABASE_PATH)

    assert len(index.law_index) == 13
    assert not index.unindexed
    law_recoveries = [item for item in index.recoveries if item.nature == "dispositivo"]
    assert len(law_recoveries) == 13


def test_database_collisions_are_not_silently_overwritten(tmp_path: Path) -> None:
    database = tmp_path / "collisions.sqlite"
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            "CREATE TABLE documentos ("
            "id INTEGER, tribunal TEXT, texto TEXT, natureza TEXT, "
            "diploma TEXT, numero TEXT)"
        )
        connection.executemany(
            "INSERT INTO documentos VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    1,
                    None,
                    "Art. 5. Texto A",
                    "dispositivo",
                    "Constituição Federal",
                    None,
                ),
                (
                    2,
                    None,
                    "Art. 5. Texto B",
                    "dispositivo",
                    "Constituição Federal",
                    None,
                ),
                (3, "STJ", "Texto A", "sumula", None, "83"),
                (4, "STJ", "Texto B", "sumula", None, "83"),
            ],
        )
        connection.commit()
    finally:
        connection.close()

    index = CanonicalMetadataIndex(database)
    assert index.law_index[("constituicao_federal", "5")] == ("1", "2")
    assert index.summary_index[("stj", "83")] == ("3", "4")
    assert index.resolve_law("artigo 5 da Constituição Federal") is None
    assert index.resolve_summary("Súmula 83 do STJ") is None
