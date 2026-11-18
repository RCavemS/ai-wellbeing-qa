"""CLI tool to classify text messages using the safety layer.

Usage:
    python -m scripts.classify_messages <path_to_messages_file>
    or:
    python scripts/classify_messages.py <path_to_messages_file>
"""

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path so src imports succeed
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.safety import SafetyLayer


def classify_file(file_path: Path) -> None:
    """Read file line-by-line and classify each message using SafetyLayer."""
    safety_layer = SafetyLayer()

    with open(file_path, "r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            message = raw_line.rstrip("\r\n")
            result = safety_layer.screen_input(message)
            print(f"{line_number} | {result.category} | {result.rule_triggered}")


def main() -> None:
    """Parse CLI arguments and run classification."""
    parser = argparse.ArgumentParser(
        description="Classify messages in a text file against the safety layer."
    )
    parser.add_argument(
        "file_path",
        type=Path,
        help="Path to text file containing one message per line",
    )
    args = parser.parse_args()

    if not args.file_path.is_file():
        print(f"Error: File not found: {args.file_path}", file=sys.stderr)
        sys.exit(1)

    classify_file(args.file_path)


if __name__ == "__main__":
    main()
