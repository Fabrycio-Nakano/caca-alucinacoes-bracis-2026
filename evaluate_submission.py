#!/usr/bin/env python3
"""Evaluate a BRACIS/Jusbrasil submission with the official local metric.

The distributed ``goldenset_offsets.csv`` has one row per citation, while the Kaggle
metric expects one row per document.  This module performs that lossless
conversion and prints the complete per-level report.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from kaggle_metric import ParticipantVisibleError, avaliar


ROOT = Path(__file__).resolve().parent


def _canonical_id(value: object) -> str:
    """Return an integer-looking canonical id without a pandas ``.0`` suffix."""
    if value is None or pd.isna(value):
        return "-"
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if not text.isdigit():
        raise ValueError(f"id_canonico inválido no goldenset: {value!r}")
    return text


def load_solution(gold_path: Path) -> pd.DataFrame:
    """Convert the public, citation-level gold file to the metric contract."""
    gold = pd.read_csv(gold_path, dtype={"documento_id": "string", "id_canonico": "string"})
    required = {
        "nivel",
        "documento_id",
        "inicio",
        "fim",
        "classificacao",
        "id_canonico",
    }
    missing = required.difference(gold.columns)
    if missing:
        raise ValueError(f"goldenset sem colunas obrigatórias: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for document_id, group in gold.groupby("documento_id", sort=False):
        levels = group["nivel"].drop_duplicates().tolist()
        if len(levels) != 1:
            raise ValueError(f"{document_id}: níveis inconsistentes: {levels}")
        blocks = []
        for row in group.sort_values(["inicio", "fim"]).itertuples(index=False):
            doc_ids = _canonical_id(row.id_canonico)
            blocks.append(
                f"{int(row.inicio)},{int(row.fim)},{row.classificacao},{doc_ids}"
            )
        rows.append(
            {
                "documento_id": str(document_id),
                "nivel": int(levels[0]),
                "citacoes": "|".join(blocks) if blocks else "-",
            }
        )
    return pd.DataFrame(rows)


def load_submission(submission_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        submission_path,
        dtype={"documento_id": "string", "citacoes": "string"},
        keep_default_na=False,
    )


def evaluate(gold_path: Path, submission_path: Path) -> dict[str, Any]:
    solution = load_solution(gold_path)
    submission = load_submission(submission_path)
    return avaliar(solution, submission, row_id="documento_id")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "submission",
        type=Path,
        help="CSV de submissão a avaliar",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=ROOT / "goldenset_offsets.csv",
        help="goldenset público (padrão: goldenset_offsets.csv ao lado deste script)",
    )
    parser.add_argument(
        "--expect",
        type=float,
        help="falha com status 1 se o score não for igual ao valor esperado",
    )
    args = parser.parse_args()

    try:
        result = evaluate(args.gold, args.submission)
    except (ParticipantVisibleError, ValueError) as exc:
        parser.exit(2, f"erro de avaliação: {exc}\n")

    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if args.expect is not None and not math.isclose(
        float(result["score_final"]), args.expect, rel_tol=0.0, abs_tol=1e-12
    ):
        parser.exit(
            1,
            f"score {result['score_final']:.12f} != esperado {args.expect:.12f}\n",
        )


if __name__ == "__main__":
    main()
