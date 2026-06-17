"""Merged placeholders for the combined classifier + biology report."""

from __future__ import annotations

from datetime import date
from typing import Any

from biology_stats import _fmt, load_biology_stats
from classifier_stats import ANALYSIS_DIR, load_classifier_stats


def load_report_stats(*, run_classifier_inference: bool = True) -> dict[str, Any]:
    """Load classifier (test split) + biology (full cohort) placeholders."""
    cache = ANALYSIS_DIR / "classifier_stats.json"
    run_clf = run_classifier_inference
    if not run_classifier_inference:
        if cache.is_file():
            run_clf = False
        else:
            raise FileNotFoundError(
                f"classifier_stats.json not found at {cache}. "
                "Run full build once (with working torch/torchvision), or: "
                "python notebooks/classifier_stats.py"
            )

    clf = load_classifier_stats(run_inference=run_clf)
    bio = load_biology_stats()
    bio_ph = bio["placeholders"]

    placeholders: dict[str, str] = {}
    placeholders.update(clf["placeholders"])
    placeholders.update(bio_ph)

    # Summary bullets combining both domains
    placeholders["summary_test_accuracy"] = clf["placeholders"]["baseline_accuracy_pct"]
    placeholders["summary_test_recall"] = clf["placeholders"]["baseline_recall_pct"]
    placeholders["summary_test_specificity"] = clf["placeholders"]["baseline_specificity_pct"]
    placeholders["summary_selective_accuracy"] = clf["placeholders"]["selective_accuracy_pct"]
    placeholders["summary_coverage_pct"] = clf["placeholders"]["coverage_pct"]

    # C6 regression (nucleus area vs cell area, first row)
    placeholders["regression_ols_slope"] = _fmt(bio.get("regression_ols_slope"), 3)
    placeholders["regression_mixed_slope"] = _fmt(bio.get("regression_mixed_slope"), 3)
    placeholders["regression_ols_r2"] = _fmt(bio.get("regression_ols_r2"), 3)
    placeholders["scaling_interpretation"] = bio_ph.get("scaling_interpretation", "")
    placeholders["report_date"] = date.today().strftime("%d %B %Y")

    # Drop biology-only large tables not used in combined report
    for key in (
        "reference_intervals_table",
        "regression_table",
        "nc_mixed_table",
        "frog_smallest_table",
        "frog_largest_table",
    ):
        placeholders.pop(key, None)

    return {
        "classifier": clf,
        "biology": bio,
        "placeholders": placeholders,
    }


if __name__ == "__main__":
    stats = load_report_stats()
    p = stats["placeholders"]
    print("n_test_cells:", p.get("n_test_cells"))
    print("n_good_cells:", p.get("n_good_cells"))
    print("area_median_um2:", p.get("area_median_um2"))
