"""Tests for scripts/classify_messages.py CLI tool."""

from pathlib import Path
import pytest
from scripts.classify_messages import classify_file, main
from src.safety import (
    CATEGORY_STANDARD,
    CATEGORY_CRISIS,
    CATEGORY_OUT_OF_SCOPE,
    RULE_NONE,
    RULE_CRISIS_HARM,
    RULE_CLINICAL_ADVICE,
)


def test_classify_file(tmp_path: Path, capsys: pytest.CaptureFixture):
    test_file = tmp_path / "sample_messages.txt"
    test_file.write_text(
        "Olá, hoje sinto-me calmo.\n"
        "Quero morrer, não aguento mais.\n"
        "Podes diagnosticar-me com depressão?\n",
        encoding="utf-8",
    )

    classify_file(test_file)

    captured = capsys.readouterr().out.splitlines()
    assert len(captured) == 3
    assert captured[0] == f"1 | {CATEGORY_STANDARD} | {RULE_NONE}"
    assert captured[1] == f"2 | {CATEGORY_CRISIS} | {RULE_CRISIS_HARM}"
    assert captured[2] == f"3 | {CATEGORY_OUT_OF_SCOPE} | {RULE_CLINICAL_ADVICE}"


def test_classify_cli_main(tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch):
    test_file = tmp_path / "sample.txt"
    test_file.write_text("Hello, how are you?\n", encoding="utf-8")

    monkeypatch.setattr("sys.argv", ["classify_messages.py", str(test_file)])
    main()

    captured = capsys.readouterr().out.strip()
    assert captured == f"1 | {CATEGORY_STANDARD} | {RULE_NONE}"


def test_classify_cli_file_not_found(tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch):
    missing_file = tmp_path / "nonexistent.txt"
    monkeypatch.setattr("sys.argv", ["classify_messages.py", str(missing_file)])

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "Error: File not found" in err
