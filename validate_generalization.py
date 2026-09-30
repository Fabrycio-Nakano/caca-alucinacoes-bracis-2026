#!/usr/bin/env python3
"""Audit the pipeline by document family without overstating generalization.

This is deliberately a *post-hoc diagnostic*, not cross-validation.  The
pipeline rules were already inspected/tuned on the public development corpus,
so slicing that same corpus into held-out-looking groups cannot turn it into an
unseen test set.  The useful question this script can answer is narrower:

* is performance concentrated in only a few document families?; and
* how much does it change when the development-only synthetic prefix/suffix
  route is explicitly enabled over the generic default?

Predictions are produced from ``txt/`` and the frozen SQLite database before
the gold file is loaded.  No canonical-ID overrides are used.  The resulting JSON
contains official metrics globally and for each pre-declared family, robust
summary statistics, provenance hashes, and an explicit leakage/limitations
section.

A clean estimate for unseen data requires freezing the predictor now and
running it once on a future corpus whose labels were unavailable during rule
development.  The current report intentionally leaves that estimate as null.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

import blind_pipeline
from evaluate_submission import load_solution
from kaggle_metric import avaliar


ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = "1.0"

# Families found in the structural audit.  They are declared rather than
# inferred from labels during this run, which keeps membership reproducible.
# They cover every public document exactly once.
TEMPLATE_FAMILIES: dict[str, tuple[str, ...]] = {
    "militar": (
        "gen_n1_001",
        "gen_n1_004",
        "gen_n2_007",
        "gen_n2_013",
    ),
    "decisao_federal": (
        "gen_n1_002",
        "gen_n2_002",
        "gen_n2_008",
    ),
    "contrarrazoes_stj": (
        "gen_n1_003",
        "gen_n1_005",
    ),
    "eleitoral": (
        "gen_n1_006",
        "gen_n1_013",
        "gen_n2_004",
        "gen_n2_006",
    ),
    "memorial": (
        "gen_n1_007",
        "gen_n1_010",
        "gen_n2_012",
    ),
    "trabalhista": (
        "gen_n1_008",
        "gen_n1_012",
        "gen_n2_003",
        "gen_n2_005",
    ),
    "agravo_civel": (
        "gen_n1_009",
        "gen_n2_011",
    ),
    "habeas_corpus": (
        "gen_n1_011",
        "gen_n2_001",
    ),
    "unicos": (
        "gen_n2_009",
        "gen_n2_010",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _corpus_sha256(text_paths: Iterable[Path]) -> str:
    """Hash names and exact bytes, independently of directory location."""

    digest = hashlib.sha256()
    for path in sorted(text_paths, key=lambda item: item.name):
        name = path.name.encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def validate_family_partition(
    document_ids: Iterable[str],
    families: Mapping[str, Iterable[str]] = TEMPLATE_FAMILIES,
) -> None:
    """Require a disjoint, exhaustive family partition."""

    expected = set(map(str, document_ids))
    owners: dict[str, str] = {}
    duplicates: dict[str, list[str]] = {}
    for family, members in families.items():
        for raw_document_id in members:
            document_id = str(raw_document_id)
            if document_id in owners:
                duplicates.setdefault(document_id, [owners[document_id]]).append(family)
            else:
                owners[document_id] = family

    assigned = set(owners)
    missing = sorted(expected - assigned)
    unexpected = sorted(assigned - expected)
    if duplicates or missing or unexpected:
        details = []
        if duplicates:
            details.append(f"duplicados={duplicates}")
        if missing:
            details.append(f"sem_familia={missing}")
        if unexpected:
            details.append(f"fora_do_gold={unexpected}")
        raise ValueError("partição de famílias inválida: " + "; ".join(details))


def predict_submission(
    text_dir: Path,
    database_path: Path,
    *,
    templates_enabled: bool,
) -> pd.DataFrame:
    """Predict every document without reading gold or enabling dev overrides."""

    text_paths = sorted(text_dir.glob("*.txt"))
    if not text_paths:
        raise ValueError(f"nenhum .txt encontrado em {text_dir}")

    resolver = blind_pipeline.CanonicalResolver(database_path)
    rows: list[dict[str, str]] = []
    for path in text_paths:
        source = blind_pipeline.read_text_exact(path)
        citations = blind_pipeline.predict_document(
            source,
            resolver,
            use_dev_templates=templates_enabled,
        )
        blind_pipeline._validate_predictions(citations, source, path.stem)
        rows.append(
            {
                "documento_id": path.stem,
                "citacoes": "|".join(item.encode() for item in citations) or "-",
            }
        )
    return pd.DataFrame(rows, columns=["documento_id", "citacoes"])


def _json_safe(value: Any) -> Any:
    """Recursively normalize numpy/pandas scalars and integer dict keys."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _evaluate_subset(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    document_ids: Iterable[str],
) -> dict[str, Any]:
    ids = set(document_ids)
    solution_subset = solution[solution["documento_id"].isin(ids)].copy()
    submission_subset = submission[submission["documento_id"].isin(ids)].copy()
    if len(solution_subset) != len(ids):
        found = set(solution_subset["documento_id"].astype(str))
        raise ValueError(f"documentos ausentes na solução: {sorted(ids - found)}")
    return _json_safe(
        avaliar(solution_subset, submission_subset, row_id="documento_id")
    )


def _support_for(
    citation_gold: pd.DataFrame,
    document_ids: Iterable[str],
) -> dict[str, Any]:
    ids = set(document_ids)
    subset = citation_gold[citation_gold["documento_id"].isin(ids)]
    by_class = subset["classificacao"].value_counts().to_dict()
    by_level = subset["nivel"].value_counts().sort_index().to_dict()
    return {
        "citations": int(len(subset)),
        "by_class": {
            label: int(by_class.get(label, 0))
            for label in ("real", "inventada", "incompleta")
        },
        "by_level": {str(int(level)): int(count) for level, count in by_level.items()},
    }


def _exact_span_metrics(
    citation_gold: pd.DataFrame,
    submission: pd.DataFrame,
    document_ids: Iterable[str],
) -> dict[str, int | float]:
    """Measure exact boundaries, which the official IoU metric can conceal."""

    ids = set(document_ids)
    gold_subset = citation_gold[citation_gold["documento_id"].isin(ids)]
    gold_spans = {
        (str(row.documento_id), int(row.inicio), int(row.fim))
        for row in gold_subset.itertuples(index=False)
    }
    predicted_spans: set[tuple[str, int, int]] = set()
    submission_subset = submission[submission["documento_id"].isin(ids)]
    for row in submission_subset.itertuples(index=False):
        cell = str(row.citacoes).strip()
        if not cell or cell == "-":
            continue
        for block in cell.split("|"):
            fields = block.split(",")
            if len(fields) != 5:
                raise ValueError(
                    f"{row.documento_id}: bloco de predição inválido: {block!r}"
                )
            predicted_spans.add((str(row.documento_id), int(fields[0]), int(fields[1])))

    true_positive = len(gold_spans & predicted_spans)
    false_positive = len(predicted_spans - gold_spans)
    false_negative = len(gold_spans - predicted_spans)
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _mode_summary(
    family_reports: Mapping[str, Mapping[str, Any]], mode: str
) -> dict[str, Any]:
    scores = {
        family: float(report[mode]["score_final"])
        for family, report in family_reports.items()
    }
    worst_family = min(scores, key=lambda family: (scores[family], family))
    return {
        # Equal family weighting deliberately prevents large/easy groups from
        # hiding a weak family.  It is diagnostic and not the Kaggle formula.
        "macro_family_score": statistics.fmean(scores.values()),
        "population_stddev": statistics.pstdev(scores.values()),
        "worst_family": worst_family,
        "worst_family_score": scores[worst_family],
        "best_family_score": max(scores.values()),
        "family_count": len(scores),
    }


def _changed_documents(
    submission_a: pd.DataFrame,
    submission_b: pd.DataFrame,
    document_ids: Iterable[str],
) -> list[str]:
    ids = set(document_ids)
    a = submission_a.set_index("documento_id")["citacoes"]
    b = submission_b.set_index("documento_id")["citacoes"]
    return sorted(
        document_id for document_id in ids if a[document_id] != b[document_id]
    )


def build_report(
    text_dir: Path,
    database_path: Path,
    gold_path: Path,
    *,
    families: Mapping[str, Iterable[str]] = TEMPLATE_FAMILIES,
) -> dict[str, Any]:
    """Build the reproducible post-hoc report.

    The order is intentional: both modes predict before ``gold_path`` is read.
    This prevents label access during this execution, although it cannot undo
    the historical tuning of source rules against this public corpus.
    """

    generic_default = predict_submission(
        text_dir, database_path, templates_enabled=False
    )
    dev_templates_on = predict_submission(
        text_dir, database_path, templates_enabled=True
    )

    # Gold is loaded only after both prediction tables are immutable.
    solution = load_solution(gold_path)
    citation_gold = pd.read_csv(
        gold_path,
        dtype={"documento_id": "string", "id_canonico": "string"},
    )
    document_ids = tuple(solution["documento_id"].astype(str))
    validate_family_partition(document_ids, families)

    expected_ids = set(document_ids)
    for mode, submission in (
        ("generic_default", generic_default),
        ("dev_templates_on", dev_templates_on),
    ):
        predicted_ids = set(submission["documento_id"].astype(str))
        if predicted_ids != expected_ids:
            raise ValueError(
                f"textos e goldenset divergem em {mode}: "
                f"sem_predicao={sorted(expected_ids - predicted_ids)}, "
                f"sem_gold={sorted(predicted_ids - expected_ids)}"
            )

    overall_generic = _evaluate_subset(solution, generic_default, document_ids)
    overall_dev_templates = _evaluate_subset(solution, dev_templates_on, document_ids)

    family_reports: dict[str, dict[str, Any]] = {}
    for family, raw_members in families.items():
        members = tuple(raw_members)
        changed = _changed_documents(generic_default, dev_templates_on, members)
        generic_result = _evaluate_subset(solution, generic_default, members)
        dev_templates_result = _evaluate_subset(solution, dev_templates_on, members)
        family_reports[family] = {
            "documents": list(members),
            "document_count": len(members),
            "evaluation_role": "post_hoc_family_slice",
            "not_a_training_holdout": True,
            "support": _support_for(citation_gold, members),
            "generic_default": generic_result,
            "dev_templates_on": dev_templates_result,
            "exact_span_diagnostics": {
                "generic_default": _exact_span_metrics(
                    citation_gold, generic_default, members
                ),
                "dev_templates_on": _exact_span_metrics(
                    citation_gold, dev_templates_on, members
                ),
            },
            "ablation": {
                "score_delta_dev_templates_minus_generic": (
                    float(dev_templates_result["score_final"])
                    - float(generic_result["score_final"])
                ),
                "changed_documents": changed,
                "changed_document_count": len(changed),
                "metric_sensitive": not math.isclose(
                    float(dev_templates_result["score_final"]),
                    float(generic_result["score_final"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
            },
        }

    all_changed = _changed_documents(generic_default, dev_templates_on, document_ids)
    predictor_paths = (
        ROOT / "blind_pipeline.py",
        ROOT / "canonical_metadata.py",
        ROOT / "template_extractor.py",
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "claim_status": "post_hoc_diagnostic_only",
        "clean_holdout": False,
        "estimated_unseen_score": None,
        "headline": {
            "generic_default_score": float(overall_generic["score_final"]),
            "dev_templates_on_score": float(overall_dev_templates["score_final"]),
            "score_delta_dev_templates_minus_generic": (
                float(overall_dev_templates["score_final"])
                - float(overall_generic["score_final"])
            ),
            "generic_default_worst_family_score": min(
                float(item["generic_default"]["score_final"])
                for item in family_reports.values()
            ),
            "dev_templates_on_worst_family_score": min(
                float(item["dev_templates_on"]["score_final"])
                for item in family_reports.values()
            ),
        },
        "methodology": {
            "name": "auditoria agrupada por família + ablação de templates de desenvolvimento",
            "family_membership": "fixed_in_source_but_defined_post_hoc",
            "partition_is_disjoint_and_exhaustive": True,
            "predictions_completed_before_gold_load": True,
            "dev_compat_enabled": False,
            "generic_default": (
                "pipeline padrão: gramáticas de identificadores e sintagmas "
                "jurídicos, sem rota de templates do desenvolvimento"
            ),
            "dev_templates_on": (
                "mesmo pipeline e resolvedor, com rota de prefixos/sufixos habilitada explicitamente"
            ),
            "primary_robustness_statistics": [
                "worst_family_score",
                "macro_family_score",
                "population_stddev",
                "score_delta_dev_templates_minus_generic",
            ],
            "official_score_note": (
                "score_final dentro de cada família usa a métrica oficial e somente "
                "os níveis presentes no grupo"
            ),
        },
        "overall": {
            "support": _support_for(citation_gold, document_ids),
            "generic_default": overall_generic,
            "dev_templates_on": overall_dev_templates,
            "exact_span_diagnostics": {
                "generic_default": _exact_span_metrics(
                    citation_gold, generic_default, document_ids
                ),
                "dev_templates_on": _exact_span_metrics(
                    citation_gold, dev_templates_on, document_ids
                ),
            },
            "ablation": {
                "score_delta_dev_templates_minus_generic": (
                    float(overall_dev_templates["score_final"])
                    - float(overall_generic["score_final"])
                ),
                "changed_documents": all_changed,
                "changed_document_count": len(all_changed),
                "metric_sensitive": not math.isclose(
                    float(overall_dev_templates["score_final"]),
                    float(overall_generic["score_final"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
            },
        },
        "family_aggregate": {
            "generic_default": _mode_summary(family_reports, "generic_default"),
            "dev_templates_on": _mode_summary(family_reports, "dev_templates_on"),
        },
        "families": family_reports,
        "leakage_audit": {
            "same_public_corpus_used_during_rule_development": True,
            "grouping_after_rule_development": True,
            "fold_specific_training_performed": False,
            "can_support_unseen_generalization_claim": False,
            "why": (
                "gramáticas jurídicas, regras semânticas e limites de span já "
                "foram ajustados após observar os 26 documentos; separar famílias "
                "agora não remove essa informação do código"
            ),
            "what_the_ablation_does_not_remove": (
                "gramáticas diretas, regras composicionais de referências "
                "incompletas, recuperação de metadados canônicos e decisões de "
                "normalização/resolução"
            ),
        },
        "clean_future_protocol": [
            "congelar os hashes do preditor e do banco antes de receber novos textos/rótulos",
            "definir famílias e critérios de aceitação sem consultar os rótulos futuros",
            "gerar e selar a submission uma única vez apenas com textos e SQLite",
            "liberar o gold somente depois; não ajustar regras com base no holdout",
            "reportar score oficial, pior família, dispersão e intervalo de incerteza por grupo",
        ],
        "limitations": [
            "A amostra contém só 26 documentos sintéticos e nove famílias; grupos têm 2 a 4 documentos.",
            "As famílias foram definidas em auditoria do corpus público e não são independentes do processo de análise.",
            "O modo genérico é uma ablação, não um modelo treinado apenas nos outros grupos.",
            "A métrica aceita IoU >= 0,5; mudanças exatas de span podem não alterar o score.",
            "O vínculo gen_n2_005/g6 foi corrigido no dataset final; não há overrides de IDs.",
            "Não há estimativa honesta do score cego até existir um conjunto realmente não observado.",
        ],
        "provenance": {
            "text_corpus_sha256": _corpus_sha256(text_dir.glob("*.txt")),
            "gold_sha256": _sha256(gold_path),
            "database_sha256": _sha256(database_path),
            "predictor_sha256": {path.name: _sha256(path) for path in predictor_paths},
            "document_count": len(document_ids),
            "family_definition": {
                family: list(members) for family, members in families.items()
            },
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texts", type=Path, default=ROOT / "txt")
    parser.add_argument("--database", type=Path, default=ROOT / "desafio1_bracis.db")
    parser.add_argument("--gold", type=Path, default=ROOT / "goldenset_offsets.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "generalization_report.json",
    )
    args = parser.parse_args()

    try:
        report = build_report(args.texts, args.database, args.gold)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"erro de validação: {exc}\n")

    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"Relatório: {args.output}")


if __name__ == "__main__":
    main()
