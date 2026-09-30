"""Tests for the family-level, post-hoc generalization audit."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_submission import load_solution  # noqa: E402
from validate_generalization import (  # noqa: E402
    TEMPLATE_FAMILIES,
    build_report,
    validate_family_partition,
)


TEXT_DIR = PROJECT_ROOT / "txt"
DATABASE_PATH = PROJECT_ROOT / "desafio1_bracis.db"
GOLD_PATH = PROJECT_ROOT / "goldenset_offsets.csv"


def test_family_partition_is_disjoint_and_exhaustive() -> None:
    solution = load_solution(GOLD_PATH)
    document_ids = solution["documento_id"].astype(str)

    validate_family_partition(document_ids)
    assert len(TEMPLATE_FAMILIES) == 9
    assert sum(map(len, TEMPLATE_FAMILIES.values())) == 26


def test_family_partition_rejects_leakage_between_groups() -> None:
    with pytest.raises(ValueError, match="duplicados"):
        validate_family_partition(
            ["doc_a"],
            {"family_a": ("doc_a",), "family_b": ("doc_a",)},
        )

    with pytest.raises(ValueError, match="sem_familia"):
        validate_family_partition(
            ["doc_a", "doc_b"],
            {"family_a": ("doc_a",)},
        )


def test_report_refuses_to_present_public_groups_as_clean_holdout() -> None:
    report = build_report(TEXT_DIR, DATABASE_PATH, GOLD_PATH)

    assert report["claim_status"] == "post_hoc_diagnostic_only"
    assert report["clean_holdout"] is False
    assert report["estimated_unseen_score"] is None
    assert report["methodology"]["dev_compat_enabled"] is False
    assert report["leakage_audit"]["can_support_unseen_generalization_claim"] is False
    assert set(report["families"]) == set(TEMPLATE_FAMILIES)

    for family_report in report["families"].values():
        assert "generic_default" in family_report
        assert "dev_templates_on" in family_report
        assert 0.0 <= family_report["generic_default"]["score_final"] <= 1.1
        assert 0.0 <= family_report["dev_templates_on"]["score_final"] <= 1.1

    # This comparison is useful even when IoU tolerance makes its score delta
    # zero: the report must retain exact-output changes as a separate signal.
    ablation = report["overall"]["ablation"]
    assert isinstance(ablation["changed_documents"], list)
    assert ablation["changed_document_count"] == len(ablation["changed_documents"])
    exact = report["overall"]["exact_span_diagnostics"]
    assert exact["generic_default"]["true_positive"] <= 192
    assert exact["dev_templates_on"]["true_positive"] <= 192
