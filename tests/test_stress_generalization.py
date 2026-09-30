"""Synthetic stress tests for citation extraction and resolution.

The examples are deliberately not copied from the development documents and
do not use their introductory/suffix templates.  They exercise task
invariants: legal surface forms, recoverable OCR noise, exact Unicode offsets,
incomplete references, header filtering, and contextual distractors.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blind_pipeline import (  # noqa: E402
    CanonicalResolver,
    extract_candidates,
    predict_document,
)


DATABASE_PATH = PROJECT_ROOT / "desafio1_bracis.db"


@dataclass(frozen=True)
class ExpectedCitation:
    literal: str
    kind: str
    explicit_type: str
    classification: str
    canonical_id: str | None = None


@pytest.fixture(scope="module")
def resolver() -> CanonicalResolver:
    """Use only the honest resolver; development overrides are out of scope."""

    result = CanonicalResolver(DATABASE_PATH)
    return result


def assert_citations(
    source: str,
    expected: list[ExpectedCitation],
    resolver: CanonicalResolver,
) -> None:
    """Assert both extraction metadata and end-to-end classifications."""

    raw_matches = extract_candidates(source)
    assert [source[item.start : item.end] for item in raw_matches] == [
        item.literal for item in expected
    ]
    assert [(item.kind, item.explicit_type) for item in raw_matches] == [
        (item.kind, item.explicit_type) for item in expected
    ]

    predictions = predict_document(source, resolver)
    assert [source[item.start : item.end] for item in predictions] == [
        item.literal for item in expected
    ]
    assert [
        (item.kind, item.classification, item.canonical_id) for item in predictions
    ] == [(item.kind, item.classification, item.canonical_id) for item in expected]

    # Offsets are Unicode-codepoint offsets into the original, unnormalized
    # text and candidates may never overlap after arbitration.
    assert all(0 <= item.start < item.end <= len(source) for item in predictions)
    assert all(
        left.end <= right.start for left, right in zip(predictions, predictions[1:])
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param(
            "Sem fórmula introdutória, a decisão adota o REsp nº 1.883.715/SP "
            "como fundamento determinante.",
            ExpectedCitation(
                "REsp nº 1.883.715/SP",
                "jurisprudencia",
                "case",
                "real",
                "1915411069",
            ),
            id="case-with-unseen-prose",
        ),
        pytest.param(
            "A distribuição do ônus segue o artigo 373, inciso I, da Lei nº "
            "13.105/2015, independentemente do rito.",
            ExpectedCitation(
                "artigo 373, inciso I, da Lei nº 13.105/2015",
                "lei",
                "law",
                "real",
                "28893055",
            ),
            id="statute-by-number",
        ),
        pytest.param(
            "A controvérsia já foi resolvida pela Súmula nº 211 do STJ e não "
            "exige novo exame.",
            ExpectedCitation(
                "Súmula nº 211 do STJ",
                "jurisprudencia",
                "sumula",
                "real",
                "1289710776",
            ),
            id="sumula-with-number-marker",
        ),
    ],
)
def test_real_references_outside_development_templates(
    source: str,
    expected: ExpectedCitation,
    resolver: CanonicalResolver,
) -> None:
    assert_citations(source, [expected], resolver)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param(
            "A peça atribui a conclusão ao REsp nº 9.999.999/SP, sem outro dado.",
            ExpectedCitation(
                "REsp nº 9.999.999/SP",
                "jurisprudencia",
                "case",
                "inventada",
            ),
            id="unknown-case",
        ),
        pytest.param(
            "O texto menciona a Súmula 999 do STJ como se fosse vinculante.",
            ExpectedCitation(
                "Súmula 999 do STJ",
                "jurisprudencia",
                "sumula",
                "inventada",
            ),
            id="unknown-sumula",
        ),
        pytest.param(
            "A obrigação estaria no art. 999 do Código Civil, segundo a parte.",
            ExpectedCitation(
                "art. 999 do Código Civil",
                "lei",
                "law",
                "inventada",
            ),
            id="unknown-statute-provision",
        ),
        pytest.param(
            "A parte atribui ao art. 373 do Código Civil uma regra inexistente.",
            ExpectedCitation(
                "art. 373 do Código Civil",
                "lei",
                "law",
                "inventada",
            ),
            id="known-article-in-wrong-statute",
        ),
    ],
)
def test_searchable_but_unknown_references_are_invented(
    source: str,
    expected: ExpectedCitation,
    resolver: CanonicalResolver,
) -> None:
    assert_citations(source, [expected], resolver)


def test_generic_jurisprudence_is_ignored_beside_explicit_references(
    resolver: CanonicalResolver,
) -> None:
    source = (
        "A defesa invoca a jurisprudência pacífica desta Corte; em reforço, "
        "cita o REsp nº 9.999.999/SP e o art. 373, I, do CPC."
    )
    assert_citations(
        source,
        [
            ExpectedCitation(
                "REsp nº 9.999.999/SP",
                "jurisprudencia",
                "case",
                "inventada",
            ),
            ExpectedCitation(
                "art. 373, I, do CPC",
                "lei",
                "law",
                "real",
                "28893055",
            ),
        ],
        resolver,
    )


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            "A decisão contrariou a jurisprudência dominante do Superior "
            "Tribunal de Justiça sobre a matéria.",
            id="unidentified-jurisprudence",
        ),
        pytest.param(
            "Devem ser observados os dispositivos aplicáveis do Código Civil "
            "antes do julgamento.",
            id="unidentified-statutory-provisions",
        ),
        pytest.param(
            "O voto remete a precedente da Terceira Turma sem indicar o número "
            "do processo.",
            id="unidentified-panel-precedent",
        ),
    ],
)
def test_generic_paraphrases_are_not_citations(
    source: str,
    resolver: CanonicalResolver,
) -> None:
    assert_citations(source, [], resolver)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param(
            "O texto digitalizado aponta o REsp n°\u00a01.883.7l5 – SP como "
            "precedente específico.",
            ExpectedCitation(
                "REsp n°\u00a01.883.7l5 – SP",
                "jurisprudencia",
                "case",
                "real",
                "1915411069",
            ),
            id="noise-in-case-number-and-separators",
        ),
        pytest.param(
            "Após revisão do OCR, consta o RE5p n°\u00a01.883.715–SP como fundamento.",
            ExpectedCitation(
                "RE5p n°\u00a01.883.715–SP",
                "jurisprudencia",
                "case",
                "real",
                "1915411069",
            ),
            id="ocr-in-procedural-prefix",
        ),
        pytest.param(
            "O reconhecimento óptico registrou Súmu1a 2ll do STJ no parágrafo.",
            ExpectedCitation(
                "Súmu1a 2ll do STJ",
                "jurisprudencia",
                "sumula",
                "real",
                "1289710776",
            ),
            id="ocr-in-sumula-word-and-number",
        ),
        pytest.param(
            "A sentença aplicou o art. 373, I, do Códig0 de Processo Civil ao "
            "caso concreto.",
            ExpectedCitation(
                "art. 373, I, do Códig0 de Processo Civil",
                "lei",
                "law",
                "real",
                "28893055",
            ),
            id="ocr-in-statute-name",
        ),
    ],
)
def test_recoverable_ocr_noise_preserves_resolution(
    source: str,
    expected: ExpectedCitation,
    resolver: CanonicalResolver,
) -> None:
    assert_citations(source, [expected], resolver)


def test_mixed_unicode_crlf_and_nbsp_keep_exact_source_offsets(
    resolver: CanonicalResolver,
) -> None:
    literal = "AgRg no Rec. Esp. n.\u00a01.522.200 (SC)"
    source = (
        "⚖️ Cabeçalho com acentuação e emoji.\r\n"
        "Na fundamentação, adotou-se o "
        f"{literal}, após exame colegiado."
    )
    expected = ExpectedCitation(
        literal,
        "jurisprudencia",
        "case",
        "real",
        "192165489",
    )

    assert_citations(source, [expected], resolver)
    prediction = predict_document(source, resolver)[0]
    assert prediction.start == source.index(literal)
    assert prediction.end == prediction.start + len(literal)


def test_own_case_identifiers_in_nonstandard_header_are_ignored(
    resolver: CanonicalResolver,
) -> None:
    # Unlike the development generator, this plausible document uses only a
    # two-newline boundary.  Both process-like values belong to the document
    # metadata; only the provision appearing after RELATÓRIO is a citation.
    source = (
        "SUPERIOR TRIBUNAL DE JUSTIÇA\n"
        "PROCESSO ORIGINÁRIO: REsp nº 1.883.715/SP\n"
        "CLASSE PROCESSUAL: HC nº 123.456/SP\n"
        "RELATOR: MINISTRO EXEMPLO\n\n"
        "RELATÓRIO\n"
        "No mérito, aplica-se o art. 373, I, do CPC."
    )
    assert_citations(
        source,
        [
            ExpectedCitation(
                "art. 373, I, do CPC",
                "lei",
                "law",
                "real",
                "28893055",
            )
        ],
        resolver,
    )


def test_same_identifier_in_header_and_body_keeps_only_body_occurrence(
    resolver: CanonicalResolver,
) -> None:
    literal = "REsp nº 1.883.715/SP"
    source = (
        f"PROCESSO: {literal}\n"
        "RELATOR: MINISTRO EXEMPLO\n\n"
        "FUNDAMENTAÇÃO\n"
        f"A controvérsia é solucionada pelo {literal}, cujo entendimento se "
        "adota."
    )
    assert_citations(
        source,
        [
            ExpectedCitation(
                literal,
                "jurisprudencia",
                "case",
                "real",
                "1915411069",
            )
        ],
        resolver,
    )
    prediction = predict_document(source, resolver)[0]
    assert prediction.start == source.rindex(literal)


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            "O hospital usa o código HC 123456 para identificar o prontuário; "
            "não se trata de processo judicial.",
            id="hospital-code",
        ),
        pytest.param(
            "O arquivo interno denominado RESP 2024 contém somente estatísticas "
            "anuais, não um precedente.",
            id="internal-report-name",
        ),
        pytest.param(
            "Inscrição OAB/SP 123.456; protocolo 1.883.715/SP; fls. 373/999; "
            "valor da causa R$ 211,00.",
            id="administrative-numbers",
        ),
    ],
)
def test_contextual_and_numeric_distractors_are_not_candidates(
    source: str,
    resolver: CanonicalResolver,
) -> None:
    assert extract_candidates(source) == []
    assert predict_document(source, resolver) == []


def test_trailing_sentence_period_is_not_part_of_numbered_law_span(
    resolver: CanonicalResolver,
) -> None:
    literal = "artigo 373, inciso I, da Lei nº 13.105/2015"
    source = f"Aplica-se o {literal}. Em seguida, examina-se a prova."
    assert_citations(
        source,
        [
            ExpectedCitation(
                literal,
                "lei",
                "law",
                "real",
                "28893055",
            )
        ],
        resolver,
    )


@pytest.mark.parametrize("literal", [
    "precedente do STJ de 2018, da relatoria de Maria Silva",
    "Reclamação do STF, de 2019, Rel. Min. Ana Souza",
])
def test_descriptive_specific_source_remains_incomplete(literal, resolver):
    source = f"A parte invoca o {literal}; pede a reforma."
    assert_citations(source, [ExpectedCitation(
        literal, "jurisprudencia", "incomplete", "incompleta"
    )], resolver)


@pytest.mark.parametrize("prefix", ["ED no AgR no AREspEl", "AgR no AREspEI", "AREspEl"])
def test_electoral_appeal_modifiers_and_ocr(prefix, resolver):
    literal = f"{prefix} 0601514-91.2020.6.05.0000"
    source = f"O voto cita {literal}; segue a fundamentação."
    result = predict_document(source, resolver)
    assert len(result) == 1
    assert source[result[0].start:result[0].end] == literal
    assert result[0].classification == "real"
    assert result[0].canonical_id == "1888419216"
