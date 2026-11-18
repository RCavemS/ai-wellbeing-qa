"""CLI tool to classify text messages using the hybrid safety pipeline (ADR-001).

Usage:
    python scripts/classify_messages.py <messages_file> --labels <labels_file>
        [--repeats 3] [--delay 5] [--stage2-only] [--out results.csv] [--yes]
"""

import argparse
import csv
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Optional, Sequence

# Ensure repository root is on sys.path so src imports succeed
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.classifier import BaseRiskClassifier, DEFAULT_MAX_ATTEMPTS, get_risk_classifier
from src.config import Config, get_config
from src.safety import (
    CATEGORY_CRISIS,
    CATEGORY_OUT_OF_SCOPE,
    CATEGORY_STANDARD,
    CATEGORY_SYSTEM_ERROR,
    SafetyLayer,
)

ALLOWED_LABELS = {"CRISIS", "CLINICAL", "NONE", "SKIP"}

CATEGORY_TO_LABEL = {
    CATEGORY_CRISIS: "CRISIS",
    CATEGORY_OUT_OF_SCOPE: "CLINICAL",
    CATEGORY_STANDARD: "NONE",
    CATEGORY_SYSTEM_ERROR: "ERROR",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Classify messages using the hybrid safety pipeline with repeats and scoring."
    )
    parser.add_argument(
        "messages_file",
        type=Path,
        help="Path to text file containing one message per line",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Path to text file containing one label per line",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of classification runs per message (default: 3)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=5.0,
        help="Seconds to wait after every call that reached Stage 2 (default: 5.0)",
    )
    parser.add_argument(
        "--stage2-only",
        action="store_true",
        default=False,
        help="Call configured classifier directly, skipping Stage 1",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Path to output CSV file for per-message table",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        default=False,
        help="Confirm execution of API calls",
    )
    return parser.parse_args(argv)


def run_classification(
    messages_path: Path,
    labels_path: Path,
    repeats: int = 3,
    delay: float = 5.0,
    stage2_only: bool = False,
    out_path: Optional[Path] = None,
    yes: bool = False,
    safety_layer: Optional[SafetyLayer] = None,
    classifier: Optional[BaseRiskClassifier] = None,
    config: Optional[Config] = None,
) -> dict[str, Any]:
    """Execute classification with repeats, scoring, and output generation."""
    if repeats < 1:
        print("Error: repeats must be at least 1.", file=sys.stderr)
        sys.exit(1)

    if not messages_path.is_file():
        print(f"Error: File not found: {messages_path}", file=sys.stderr)
        sys.exit(1)

    if not labels_path.is_file():
        print(f"Error: File not found: {labels_path}", file=sys.stderr)
        sys.exit(1)

    with open(messages_path, "r", encoding="utf-8") as f:
        messages = [line.rstrip("\r\n") for line in f]

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f]

    if len(messages) != len(labels):
        print(
            f"Error: Line count mismatch: {messages_path} has {len(messages)} lines, but {labels_path} has {len(labels)} lines.",
            file=sys.stderr,
        )
        sys.exit(1)

    for idx, label in enumerate(labels, start=1):
        if label not in ALLOWED_LABELS:
            print(
                f"Error: Invalid label '{label}' on line {idx}. Allowed labels: CRISIS, CLINICAL, NONE, SKIP.",
                file=sys.stderr,
            )
            sys.exit(1)

    if config is None:
        config = get_config(reload=True)

    provider = (config.classifier_provider or "").lower().strip()
    if provider != "gemini":
        print(
            f"Error: CLASSIFIER_PROVIDER must be 'gemini', but found '{config.classifier_provider}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    active_classifier = classifier or (safety_layer.classifier if safety_layer is not None else None)
    max_attempts = getattr(active_classifier, "max_attempts", DEFAULT_MAX_ATTEMPTS)
    max_classifications = len(messages) * repeats
    max_http_requests = max_classifications * max_attempts
    print(
        f"Maximum number of API calls: {max_classifications} classifications ({len(messages)} messages × {repeats} repeats), "
        f"up to {max_http_requests} HTTP requests (max_attempts={max_attempts})"
    )

    if not yes:
        print("Stopping: run with --yes to proceed.", file=sys.stderr)
        sys.exit(1)

    if classifier is None:
        if safety_layer is not None:
            classifier = safety_layer.classifier
        else:
            classifier = get_risk_classifier(config)

    if not stage2_only and safety_layer is None:
        safety_layer = SafetyLayer(classifier=classifier)

    initial_request_count = getattr(classifier, "request_count", 0)
    per_message_results: list[dict[str, Any]] = []

    for line_num, (message, expected_label) in enumerate(zip(messages, labels), start=1):
        runs: list[dict[str, Any]] = []
        for _ in range(repeats):
            if stage2_only:
                t0 = time.perf_counter()
                try:
                    raw_label = classifier.classify(message)
                    latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                    label = raw_label
                    stage = "STAGE_2"
                    error_type = None
                except Exception as exc:
                    latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                    label = "ERROR"
                    stage = "STAGE_2"
                    error_type = getattr(exc, "error_type", type(exc).__name__)

                if delay > 0:
                    time.sleep(delay)
            else:
                assert safety_layer is not None
                result = safety_layer.screen_input(message)
                label = CATEGORY_TO_LABEL.get(result.category, result.category)
                stage = result.stage_decided
                latency_ms = result.latency_ms
                error_type = result.error_type

                reached_stage_2 = not (stage == "STAGE_1" and label == "CRISIS")
                if reached_stage_2:
                    if delay > 0:
                        time.sleep(delay)

            runs.append(
                {
                    "label": label,
                    "stage": stage,
                    "latency_ms": latency_ms,
                    "error_type": error_type,
                }
            )

        all_agreed = len(set(r["label"] for r in runs)) == 1
        per_message_results.append(
            {
                "line": line_num,
                "expected": expected_label,
                "runs": runs,
                "agreed": all_agreed,
            }
        )

    api_calls_made = getattr(classifier, "request_count", 0) - initial_request_count

    # Output per-message table (never print or save message text)
    print("\n--- Per-Message Classification Results ---")
    for item in per_message_results:
        line_num = item["line"]
        exp = item["expected"]
        agreed = item["agreed"]
        runs_str = " | ".join(
            f"run {idx}: {r['label']} ({r['stage']})"
            for idx, r in enumerate(item["runs"], start=1)
        )
        print(f"line {line_num} | expected: {exp} | {runs_str} | agreed: {agreed}")

    # Summary calculations
    label_counts = {lbl: 0 for lbl in ["CRISIS", "CLINICAL", "NONE", "SKIP"]}
    for lbl in labels:
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    skip_lines = [item["line"] for item in per_message_results if item["expected"] == "SKIP"]
    eval_items = [item for item in per_message_results if item["expected"] != "SKIP"]

    crisis_items = [item for item in eval_items if item["expected"] == "CRISIS"]
    detected_crisis = sum(
        1 for item in crisis_items if all(r["label"] == "CRISIS" for r in item["runs"])
    )
    total_crisis = len(crisis_items)

    clinical_items = [item for item in eval_items if item["expected"] == "CLINICAL"]
    detected_clinical = sum(
        1 for item in clinical_items if all(r["label"] == "CLINICAL" for r in item["runs"])
    )
    total_clinical = len(clinical_items)

    none_items = [item for item in eval_items if item["expected"] == "NONE"]
    total_none = len(none_items)

    crisis_fp_lines = [
        item["line"] for item in none_items if any(r["label"] == "CRISIS" for r in item["runs"])
    ]
    any_flag_fp_lines = [
        item["line"] for item in none_items if any(r["label"] != "NONE" for r in item["runs"])
    ]

    error_occurrences = []
    grouped_errors: dict[str, list[int]] = {}
    for item in eval_items:
        line = item["line"]
        for r in item["runs"]:
            if r["label"] == "ERROR":
                err_type = r["error_type"] or "UnknownError"
                error_occurrences.append((err_type, line))
                if err_type not in grouped_errors:
                    grouped_errors[err_type] = []
                if line not in grouped_errors[err_type]:
                    grouped_errors[err_type].append(line)

    agreed_eval = sum(1 for item in eval_items if item["agreed"])
    total_eval = len(eval_items)

    all_latencies = [r["latency_ms"] for item in eval_items for r in item["runs"]]
    med_latency = statistics.median(all_latencies) if all_latencies else 0.0

    print("\n=== SUMMARY ===")
    print("Label counts in labels file:")
    for lbl in ["CRISIS", "CLINICAL", "NONE", "SKIP"]:
        print(f"  {lbl}: {label_counts[lbl]}")
    print(f"  Total: {len(labels)}")

    if skip_lines:
        print(f"\nSKIP lines (excluded from metrics): {skip_lines}")
    else:
        print("\nSKIP lines (excluded from metrics): none")

    if total_crisis > 0:
        pct = (detected_crisis / total_crisis) * 100.0
        print(f"\nCrisis recall: {detected_crisis}/{total_crisis} ({pct:.1f}%)")
    else:
        print("\nCrisis recall: N/A (0 CRISIS messages)")

    if total_clinical > 0:
        pct = (detected_clinical / total_clinical) * 100.0
        print(f"Clinical detection: {detected_clinical}/{total_clinical} ({pct:.1f}%)")
    else:
        print("Clinical detection: N/A (0 CLINICAL messages)")

    if total_none > 0:
        crisis_fp_pct = (len(crisis_fp_lines) / total_none) * 100.0
        print(f"Crisis false positives: {len(crisis_fp_lines)}/{total_none} ({crisis_fp_pct:.1f}%)")
        if crisis_fp_lines:
            print(f"  Affected lines: {crisis_fp_lines}")

        any_fp_pct = (len(any_flag_fp_lines) / total_none) * 100.0
        print(f"Any-flag false positives: {len(any_flag_fp_lines)}/{total_none} ({any_fp_pct:.1f}%)")
        if any_flag_fp_lines:
            print(f"  Affected lines: {any_flag_fp_lines}")
    else:
        print("Crisis false positives: N/A (0 NONE messages)")
        print("Any-flag false positives: N/A (0 NONE messages)")

    if not grouped_errors:
        print("\nErrors: 0")
    else:
        print(f"\nErrors ({len(error_occurrences)} total occurrence(s) across evaluated lines):")
        for err_type, lines in grouped_errors.items():
            print(f"  - {err_type}: {len(lines)} message(s) affected on line(s) {lines}")

    if total_eval > 0:
        agr_pct = (agreed_eval / total_eval) * 100.0
        print(f"\nRun agreement rate: {agreed_eval}/{total_eval} ({agr_pct:.1f}%)")
    else:
        print("\nRun agreement rate: N/A")

    if all_latencies:
        print(f"Median latency: {med_latency:.1f} ms")
    else:
        print("Median latency: N/A")

    print(f"API calls made: {api_calls_made}")

    # Output CSV table (never print or save message text)
    if out_path is not None:
        header = ["line", "expected"]
        for r in range(1, repeats + 1):
            header.extend([f"run_{r}", f"stage_{r}"])
        header.append("agreed")

        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for item in per_message_results:
                row: list[Any] = [item["line"], item["expected"]]
                for r in item["runs"]:
                    row.extend([r["label"], r["stage"]])
                row.append(item["agreed"])
                writer.writerow(row)

    return {
        "per_message_results": per_message_results,
        "label_counts": label_counts,
        "skip_lines": skip_lines,
        "crisis_recall": (detected_crisis, total_crisis),
        "clinical_detection": (detected_clinical, total_clinical),
        "crisis_false_positives": (len(crisis_fp_lines), total_none),
        "any_flag_false_positives": (len(any_flag_fp_lines), total_none),
        "grouped_errors": grouped_errors,
        "run_agreement_rate": (agreed_eval, total_eval),
        "median_latency": med_latency,
        "api_calls_made": api_calls_made,
    }


def classify_file(
    file_path: Path,
    labels_path: Optional[Path] = None,
    repeats: int = 3,
    delay: float = 5.0,
    stage2_only: bool = False,
    out_path: Optional[Path] = None,
    yes: bool = False,
    safety_layer: Optional[SafetyLayer] = None,
    classifier: Optional[BaseRiskClassifier] = None,
    config: Optional[Config] = None,
) -> dict[str, Any]:
    """Compatibility wrapper calling run_classification."""
    if labels_path is None:
        raise ValueError("labels_path is required.")
    return run_classification(
        messages_path=file_path,
        labels_path=labels_path,
        repeats=repeats,
        delay=delay,
        stage2_only=stage2_only,
        out_path=out_path,
        yes=yes,
        safety_layer=safety_layer,
        classifier=classifier,
        config=config,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse CLI arguments and run classification."""
    args = parse_args(argv)
    run_classification(
        messages_path=args.messages_file,
        labels_path=args.labels,
        repeats=args.repeats,
        delay=args.delay,
        stage2_only=args.stage2_only,
        out_path=args.out,
        yes=args.yes,
    )


if __name__ == "__main__":
    main()
