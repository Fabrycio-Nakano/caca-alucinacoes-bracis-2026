#!/usr/bin/env python3
"""Build the public-development oracle submission.

This command is deliberately isolated from the blind inference pipeline.  It
uses the distributed development gold labels to test offsets, JSON/CSV
serialization and the official metric end to end.  It must never be used for
the future blind evaluation set.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from blind_pipeline import read_text_exact
from json_to_submission import encode


ROOT = Path(__file__).resolve().parent


def _clean_id(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if not text.isdigit():
        raise ValueError(f"id_canonico inválido: {value!r}")
    return text


def build_oracle(
    gold_path: Path,
    text_dir: Path,
    json_dir: Path,
    submission_path: Path,
) -> None:
    gold = pd.read_csv(
        gold_path,
        dtype={"documento_id": "string", "id_canonico": "string"},
    )
    json_dir.mkdir(parents=True, exist_ok=True)

    input_paths = sorted(text_dir.glob("*.txt"))
    if not input_paths:
        raise ValueError(f"nenhum .txt encontrado em {text_dir}")
    groups = {
        str(document_id): group
        for document_id, group in gold.groupby("documento_id", sort=False)
    }
    expected_documents = [path.stem for path in input_paths]
    extra_gold = sorted(set(groups).difference(expected_documents))
    if extra_gold:
        raise ValueError(f"goldenset contém documentos sem TXT: {extra_gold}")
    stale_json = sorted(
        path.name
        for path in json_dir.glob("*.json")
        if path.stem not in expected_documents
    )
    if stale_json:
        raise ValueError(
            f"{json_dir} contém JSONs de outra execução: {stale_json}; "
            "use uma pasta de saída vazia para não misturar conjuntos"
        )

    submission_rows: list[tuple[str, str]] = []
    for source_path in input_paths:
        document_id = source_path.stem
        source = read_text_exact(source_path)
        citations = []
        group = groups.get(document_id)
        if group is not None:
            for row in group.sort_values(["inicio", "fim"]).itertuples(index=False):
                start, end = int(row.inicio), int(row.fim)
                literal = source[start:end]
                expected = str(row.trecho).replace("\\n", "\n")
                if literal != expected:
                    raise ValueError(
                        f"{document_id}/{row.citacao_id}: trecho não corresponde ao span "
                        f"[{start}, {end})"
                    )
                canonical_id = _clean_id(row.id_canonico)
                resolution = (
                    {"id_canonico": canonical_id}
                    if str(row.classificacao) == "real"
                    else None
                )
                citations.append(
                    {
                        "citacao_id": str(row.citacao_id),
                        "inicio": start,
                        "fim": end,
                        "trecho": literal,
                        "tipo": str(row.tipo),
                        "classificacao": str(row.classificacao),
                        "resolucao": resolution,
                        "confianca": 1.0,
                    }
                )

        payload = {
            "schema_version": "1.2",
            "documento_id": str(document_id),
            "citacoes": citations,
        }
        (json_dir / f"{document_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        # Re-read what was actually serialized and pass it through the official
        # converter function.  JSON and CSV are therefore one end-to-end path.
        serialized = json.loads(
            (json_dir / f"{document_id}.json").read_text(encoding="utf-8")
        )
        submission_rows.append((str(document_id), encode(serialized)))

    generated_documents = sorted(document_id for document_id, _ in submission_rows)
    if generated_documents != sorted(expected_documents):
        missing = sorted(set(expected_documents) - set(generated_documents))
        extra = sorted(set(generated_documents) - set(expected_documents))
        raise ValueError(f"cobertura de documentos inválida; faltando={missing}, extras={extra}")

    submission_path.parent.mkdir(parents=True, exist_ok=True)
    with submission_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["documento_id", "citacoes"])
        writer.writerows(submission_rows)

    print(f"JSONs: {json_dir} ({len(submission_rows)} documentos)")
    print(f"Submission: {submission_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=ROOT / "goldenset_offsets.csv")
    parser.add_argument("--texts", type=Path, default=ROOT / "txt")
    parser.add_argument(
        "--json-dir", type=Path, default=ROOT / "artifacts" / "dev_oracle_json"
    )
    parser.add_argument(
        "--submission",
        type=Path,
        default=ROOT / "artifacts" / "submission_dev_1_1000.csv",
    )
    args = parser.parse_args()
    build_oracle(args.gold, args.texts, args.json_dir, args.submission)


if __name__ == "__main__":
    main()
