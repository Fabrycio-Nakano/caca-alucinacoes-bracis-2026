#!/usr/bin/env python3
"""Run and verify the complete development pipeline in one command."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from blind_pipeline import run_pipeline
from build_dev_oracle import build_oracle
from evaluate_submission import evaluate


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
TARGET_SCORE = 1.1


def _require_target(name: str, report: dict[str, Any]) -> None:
    score = float(report["score_final"])
    if not math.isclose(score, TARGET_SCORE, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"{name}: score {score:.12f}, meta {TARGET_SCORE:.12f}")


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    blind_submission = ARTIFACTS / "submission_blind.csv"
    oracle_submission = ARTIFACTS / "submission_oracle.csv"

    run_pipeline(
        ROOT / "txt",
        ROOT / "desafio1_bracis.db",
        ARTIFACTS / "blind_json",
        blind_submission,
    )
    blind_report = evaluate(ROOT / "goldenset_offsets.csv", blind_submission)

    build_oracle(
        ROOT / "goldenset_offsets.csv",
        ROOT / "txt",
        ARTIFACTS / "dev_oracle_json",
        oracle_submission,
    )
    oracle_report = evaluate(ROOT / "goldenset_offsets.csv", oracle_submission)

    _require_target("generic-final", blind_report)
    _require_target("oracle-dev", oracle_report)
    report = {
        "target": TARGET_SCORE,
        "blind": blind_report,
        "oracle_dev": oracle_report,
        "artifacts": {
            "blind_submission": str(blind_submission),
            "oracle_submission": str(oracle_submission),
        },
    }
    report_path = ARTIFACTS / "evaluation_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Relatório: {report_path}")


if __name__ == "__main__":
    main()
