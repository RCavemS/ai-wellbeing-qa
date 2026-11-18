"""Tests for scripts/classify_messages.py CLI tool (Issue #33, ADR-001)."""

import csv
from pathlib import Path
import pytest
from scripts.classify_messages import classify_file, main, run_classification
from src.classifier import ClassifierError, MockRiskClassifier
from src.config import Config


def test_line_count_mismatch(tmp_path: Path, capsys: pytest.CaptureFixture):
    msg_file = tmp_path / "messages.txt"
    msg_file.write_text("msg 1\nmsg 2\nmsg 3\n", encoding="utf-8")
    lbl_file = tmp_path / "labels.txt"
    lbl_file.write_text("CRISIS\nNONE\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        main([str(msg_file), "--labels", str(lbl_file), "--yes"])

    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "Line count mismatch" in err


def test_invalid_label(tmp_path: Path, capsys: pytest.CaptureFixture):
    msg_file = tmp_path / "messages.txt"
    msg_file.write_text("msg 1\n", encoding="utf-8")
    lbl_file = tmp_path / "labels.txt"
    lbl_file.write_text("INVALID_TAG\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        main([str(msg_file), "--labels", str(lbl_file), "--yes"])

    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "Invalid label 'INVALID_TAG'" in err


def test_repeats_less_than_one(tmp_path: Path, capsys: pytest.CaptureFixture):
    msg_file = tmp_path / "messages.txt"
    msg_file.write_text("msg 1\n", encoding="utf-8")
    lbl_file = tmp_path / "labels.txt"
    lbl_file.write_text("NONE\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        main([str(msg_file), "--labels", str(lbl_file), "--repeats", "0", "--yes"])

    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "repeats must be at least 1" in err


def test_refuses_to_run_with_provider_mock(
    tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
):
    msg_file = tmp_path / "messages.txt"
    msg_file.write_text("msg 1\n", encoding="utf-8")
    lbl_file = tmp_path / "labels.txt"
    lbl_file.write_text("NONE\n", encoding="utf-8")

    monkeypatch.setenv("CLASSIFIER_PROVIDER", "mock")

    with pytest.raises(SystemExit) as exc_info:
        main([str(msg_file), "--labels", str(lbl_file), "--yes"])

    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "CLASSIFIER_PROVIDER must be 'gemini'" in err


def test_refuses_without_yes(
    tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
):
    msg_file = tmp_path / "messages.txt"
    msg_file.write_text("msg 1\nmsg 2\n", encoding="utf-8")
    lbl_file = tmp_path / "labels.txt"
    lbl_file.write_text("NONE\nNONE\n", encoding="utf-8")

    monkeypatch.setenv("CLASSIFIER_PROVIDER", "gemini")

    with pytest.raises(SystemExit) as exc_info:
        main([str(msg_file), "--labels", str(lbl_file), "--repeats", "3"])

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Maximum number of API calls: 6 classifications (2 messages × 3 repeats), up to 18 HTTP requests (max_attempts=3)" in captured.out
    assert "Stopping: run with --yes to proceed." in captured.err


def test_recall_uses_every_run_rule(tmp_path: Path, capsys: pytest.CaptureFixture):
    msg_file = tmp_path / "messages.txt"
    msg_file.write_text("crisis msg 1\ncrisis msg 2\n", encoding="utf-8")
    lbl_file = tmp_path / "labels.txt"
    lbl_file.write_text("CRISIS\nCRISIS\n", encoding="utf-8")

    # Message 1: 2 CRISIS then 1 NONE -> missed!
    # Message 2: 3 CRISIS -> detected!
    responses = [
        "CRISIS",
        "CRISIS",
        "NONE",
        "CRISIS",
        "CRISIS",
        "CRISIS",
    ]
    mock_classifier = MockRiskClassifier(responses=responses)
    config = Config(classifier_provider="gemini")

    result = run_classification(
        messages_path=msg_file,
        labels_path=lbl_file,
        repeats=3,
        delay=0,
        yes=True,
        classifier=mock_classifier,
        config=config,
    )

    assert result["crisis_recall"] == (1, 2)
    captured = capsys.readouterr().out
    assert "Crisis recall: 1/2 (50.0%)" in captured
    assert "line 1 | expected: CRISIS | run 1: CRISIS (STAGE_2) | run 2: CRISIS (STAGE_2) | run 3: NONE (STAGE_2) | agreed: False" in captured
    assert "line 2 | expected: CRISIS | run 1: CRISIS (STAGE_2) | run 2: CRISIS (STAGE_2) | run 3: CRISIS (STAGE_2) | agreed: True" in captured


def test_clinical_detection_uses_every_run_rule(tmp_path: Path, capsys: pytest.CaptureFixture):
    msg_file = tmp_path / "messages.txt"
    msg_file.write_text("clinical msg 1\nclinical msg 2\n", encoding="utf-8")
    lbl_file = tmp_path / "labels.txt"
    lbl_file.write_text("CLINICAL\nCLINICAL\n", encoding="utf-8")

    # Message 1: 3 CLINICAL -> detected
    # Message 2: 2 CLINICAL then 1 NONE -> missed
    responses = [
        "CLINICAL",
        "CLINICAL",
        "CLINICAL",
        "CLINICAL",
        "CLINICAL",
        "NONE",
    ]
    mock_classifier = MockRiskClassifier(responses=responses)
    config = Config(classifier_provider="gemini")

    result = run_classification(
        messages_path=msg_file,
        labels_path=lbl_file,
        repeats=3,
        delay=0,
        yes=True,
        classifier=mock_classifier,
        config=config,
    )

    assert result["clinical_detection"] == (1, 2)
    captured = capsys.readouterr().out
    assert "Clinical detection: 1/2 (50.0%)" in captured


def test_error_results_count_as_not_detected_and_are_listed(
    tmp_path: Path, capsys: pytest.CaptureFixture
):
    msg_file = tmp_path / "messages.txt"
    msg_file.write_text("crisis msg\n", encoding="utf-8")
    lbl_file = tmp_path / "labels.txt"
    lbl_file.write_text("CRISIS\n", encoding="utf-8")

    # Run 1: CRISIS, Run 2: TimeoutError, Run 3: CRISIS -> missed!
    responses = [
        "CRISIS",
        ClassifierError("Request timed out", error_type="TimeoutError"),
        "CRISIS",
    ]
    mock_classifier = MockRiskClassifier(responses=responses)
    config = Config(classifier_provider="gemini")

    result = run_classification(
        messages_path=msg_file,
        labels_path=lbl_file,
        repeats=3,
        delay=0,
        yes=True,
        classifier=mock_classifier,
        config=config,
    )

    assert result["crisis_recall"] == (0, 1)
    assert "TimeoutError" in result["grouped_errors"]
    assert result["grouped_errors"]["TimeoutError"] == [1]

    captured = capsys.readouterr().out
    assert "Crisis recall: 0/1 (0.0%)" in captured
    assert "TimeoutError: 1 message(s) affected on line(s) [1]" in captured


def test_skip_excluded_from_metrics(tmp_path: Path, capsys: pytest.CaptureFixture):
    msg_file = tmp_path / "messages.txt"
    msg_file.write_text(
        "msg 1 crisis\n"
        "msg 2 none\n"
        "msg 3 skip with error\n"
        "msg 4 skip with crisis\n",
        encoding="utf-8",
    )
    lbl_file = tmp_path / "labels.txt"
    lbl_file.write_text("CRISIS\nNONE\nSKIP\nSKIP\n", encoding="utf-8")

    # 4 messages × 2 repeats = 8 responses
    responses = [
        "CRISIS",
        "CRISIS",  # Line 1: CRISIS -> detected
        "NONE",
        "NONE",  # Line 2: NONE -> agreed
        ClassifierError("Skip error", error_type="SkipTimeout"),
        "CRISIS",  # Line 3: SKIP
        "CRISIS",
        "CRISIS",  # Line 4: SKIP
    ]
    mock_classifier = MockRiskClassifier(responses=responses)
    config = Config(classifier_provider="gemini")

    result = run_classification(
        messages_path=msg_file,
        labels_path=lbl_file,
        repeats=2,
        delay=0,
        yes=True,
        classifier=mock_classifier,
        config=config,
    )

    assert result["skip_lines"] == [3, 4]
    assert result["crisis_recall"] == (1, 1)
    assert result["crisis_false_positives"] == (0, 1)
    assert result["any_flag_false_positives"] == (0, 1)
    assert result["run_agreement_rate"] == (2, 2)
    assert "SkipTimeout" not in result["grouped_errors"]

    captured = capsys.readouterr().out
    assert "SKIP lines (excluded from metrics): [3, 4]" in captured
    assert "Crisis recall: 1/1 (100.0%)" in captured
    assert "Crisis false positives: 0/1 (0.0%)" in captured
    assert "Run agreement rate: 2/2 (100.0%)" in captured


def test_no_message_text_appears_in_stdout_or_csv(tmp_path: Path, capsys: pytest.CaptureFixture):
    secret_text_1 = "SUPER_SECRET_USER_INPUT_ALPHA"
    secret_text_2 = "CONFIDENTIAL_USER_DISCLOSURE_BETA"

    msg_file = tmp_path / "messages.txt"
    msg_file.write_text(f"{secret_text_1}\n{secret_text_2}\n", encoding="utf-8")
    lbl_file = tmp_path / "labels.txt"
    lbl_file.write_text("NONE\nNONE\n", encoding="utf-8")
    out_csv = tmp_path / "results.csv"

    mock_classifier = MockRiskClassifier(default_label="NONE")
    config = Config(classifier_provider="gemini")

    run_classification(
        messages_path=msg_file,
        labels_path=lbl_file,
        repeats=2,
        delay=0,
        out_path=out_csv,
        yes=True,
        classifier=mock_classifier,
        config=config,
    )

    captured = capsys.readouterr()
    stdout_text = captured.out
    stderr_text = captured.err
    csv_text = out_csv.read_text(encoding="utf-8")

    assert secret_text_1 not in stdout_text
    assert secret_text_2 not in stdout_text
    assert secret_text_1 not in stderr_text
    assert secret_text_2 not in stderr_text
    assert secret_text_1 not in csv_text
    assert secret_text_2 not in csv_text


def test_stage2_only_never_calls_stage1(tmp_path: Path, capsys: pytest.CaptureFixture):
    # This text matches Stage 1 regex (_CRISIS_PATTERNS)
    msg_file = tmp_path / "messages.txt"
    msg_file.write_text("Quero morrer, não aguento mais.\n", encoding="utf-8")
    lbl_file = tmp_path / "labels.txt"
    lbl_file.write_text("NONE\n", encoding="utf-8")

    # In --stage2-only mode, Stage 1 is bypassed completely; the classifier returns NONE
    mock_classifier = MockRiskClassifier(default_label="NONE")
    config = Config(classifier_provider="gemini")

    result = run_classification(
        messages_path=msg_file,
        labels_path=lbl_file,
        repeats=2,
        delay=0,
        stage2_only=True,
        yes=True,
        classifier=mock_classifier,
        config=config,
    )

    runs = result["per_message_results"][0]["runs"]
    for r in runs:
        assert r["stage"] == "STAGE_2"
        assert r["label"] == "NONE"

    captured = capsys.readouterr().out
    assert "STAGE_1" not in captured
    assert "run 1: NONE (STAGE_2)" in captured


def test_false_positives_calculation(tmp_path: Path, capsys: pytest.CaptureFixture):
    msg_file = tmp_path / "messages.txt"
    msg_file.write_text("benign 1\nbenign 2\nbenign 3\n", encoding="utf-8")
    lbl_file = tmp_path / "labels.txt"
    lbl_file.write_text("NONE\nNONE\nNONE\n", encoding="utf-8")

    # Line 1: [NONE, NONE] -> benign
    # Line 2: [NONE, CRISIS] -> crisis FP and any-flag FP
    # Line 3: [NONE, CLINICAL] -> any-flag FP only
    responses = [
        "NONE",
        "NONE",
        "NONE",
        "CRISIS",
        "NONE",
        "CLINICAL",
    ]
    mock_classifier = MockRiskClassifier(responses=responses)
    config = Config(classifier_provider="gemini")

    result = run_classification(
        messages_path=msg_file,
        labels_path=lbl_file,
        repeats=2,
        delay=0,
        yes=True,
        classifier=mock_classifier,
        config=config,
    )

    assert result["crisis_false_positives"] == (1, 3)
    assert result["any_flag_false_positives"] == (2, 3)

    captured = capsys.readouterr().out
    assert "Crisis false positives: 1/3 (33.3%)" in captured
    assert "Any-flag false positives: 2/3 (66.7%)" in captured


def test_csv_output_format(tmp_path: Path):
    msg_file = tmp_path / "messages.txt"
    msg_file.write_text("msg 1\nmsg 2\n", encoding="utf-8")
    lbl_file = tmp_path / "labels.txt"
    lbl_file.write_text("CRISIS\nNONE\n", encoding="utf-8")
    out_csv = tmp_path / "results.csv"

    responses = ["CRISIS", "CRISIS", "NONE", "NONE"]
    mock_classifier = MockRiskClassifier(responses=responses)
    config = Config(classifier_provider="gemini")

    run_classification(
        messages_path=msg_file,
        labels_path=lbl_file,
        repeats=2,
        delay=0,
        out_path=out_csv,
        yes=True,
        classifier=mock_classifier,
        config=config,
    )

    with open(out_csv, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))

    assert reader[0] == ["line", "expected", "run_1", "stage_1", "run_2", "stage_2", "agreed"]
    assert reader[1] == ["1", "CRISIS", "CRISIS", "STAGE_2", "CRISIS", "STAGE_2", "True"]
    assert reader[2] == ["2", "NONE", "NONE", "STAGE_2", "NONE", "STAGE_2", "True"]


def test_classify_cli_file_not_found(tmp_path: Path, capsys: pytest.CaptureFixture):
    missing_file = tmp_path / "nonexistent.txt"
    lbl_file = tmp_path / "labels.txt"
    lbl_file.write_text("NONE\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        main([str(missing_file), "--labels", str(lbl_file), "--yes"])
    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "Error: File not found" in err


def test_classify_file_backward_compatibility(tmp_path: Path):
    msg_file = tmp_path / "messages.txt"
    msg_file.write_text("Olá, sinto-me bem.\n", encoding="utf-8")
    lbl_file = tmp_path / "labels.txt"
    lbl_file.write_text("NONE\n", encoding="utf-8")

    mock_classifier = MockRiskClassifier(default_label="NONE")
    config = Config(classifier_provider="gemini")

    res = classify_file(
        file_path=msg_file,
        labels_path=lbl_file,
        repeats=1,
        delay=0,
        yes=True,
        classifier=mock_classifier,
        config=config,
    )

    assert res["run_agreement_rate"] == (1, 1)

    with pytest.raises(ValueError) as exc:
        classify_file(msg_file)
    assert "labels_path is required" in str(exc.value)


def test_stage1_crisis_skips_stage2_and_delay(
    tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
):
    msg_file = tmp_path / "messages.txt"
    msg_file.write_text("Quero morrer, não aguento mais.\n", encoding="utf-8")
    lbl_file = tmp_path / "labels.txt"
    lbl_file.write_text("CRISIS\n", encoding="utf-8")

    # Mock classifier should not be called at all because Stage 1 intercepts crisis
    mock_classifier = MockRiskClassifier()
    config = Config(classifier_provider="gemini")

    sleep_calls: list[float] = []
    monkeypatch.setattr("time.sleep", lambda s: sleep_calls.append(s))

    result = run_classification(
        messages_path=msg_file,
        labels_path=lbl_file,
        repeats=3,
        delay=5.0,
        yes=True,
        classifier=mock_classifier,
        config=config,
    )

    assert mock_classifier.call_count == 0
    assert result["api_calls_made"] == 0
    assert len(sleep_calls) == 0  # No wait after Stage 1 crisis match!

    captured = capsys.readouterr().out
    assert "run 1: CRISIS (STAGE_1)" in captured
    assert "API calls made: 0" in captured


def test_classify_messages_counts_real_http_requests_on_retry_and_zero_on_credential_error(
    tmp_path: Path, capsys: pytest.CaptureFixture
):
    """Issue #35: 429-then-200 sequence counts 2 HTTP requests; credential error counts 0 HTTP requests."""
    from unittest.mock import MagicMock, patch
    import httpx
    from src.classifier import GeminiRiskClassifier

    msg_file = tmp_path / "messages.txt"
    msg_file.write_text("Hoje tive um dia tranquilo.\n", encoding="utf-8")
    lbl_file = tmp_path / "labels.txt"
    lbl_file.write_text("NONE\n", encoding="utf-8")
    config = Config(classifier_provider="gemini")

    # 1. 429-then-200 sequence on 1 classification (repeats=1) -> 2 HTTP requests
    resp_429 = MagicMock(spec=httpx.Response)
    resp_429.status_code = 429
    resp_200 = MagicMock(spec=httpx.Response)
    resp_200.status_code = 200
    resp_200.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": '{"label": "NONE"}'}]}}]
    }

    gemini_clf = GeminiRiskClassifier(
        api_key="fake-api-key",
        model_name="gemini-3.5-flash-lite",
        max_attempts=3,
        backoff_base=0.0,
    )

    with patch("httpx.post", side_effect=[resp_429, resp_200]) as mock_post:
        res_retry = run_classification(
            messages_path=msg_file,
            labels_path=lbl_file,
            repeats=1,
            delay=0,
            yes=True,
            classifier=gemini_clf,
            config=config,
        )

    assert mock_post.call_count == 2
    assert gemini_clf.request_count == 2
    assert res_retry["api_calls_made"] == 2
    captured_retry = capsys.readouterr().out
    assert "Maximum number of API calls: 1 classifications (1 messages × 1 repeats), up to 3 HTTP requests (max_attempts=3)" in captured_retry
    assert "API calls made: 2" in captured_retry

    # 2. Credential error before any request -> 0 HTTP requests
    gemini_bad_creds = GeminiRiskClassifier(
        api_key="",
        model_name="gemini-3.5-flash-lite",
        max_attempts=3,
    )

    with patch("httpx.post") as mock_post_no_call:
        res_cred_err = run_classification(
            messages_path=msg_file,
            labels_path=lbl_file,
            repeats=1,
            delay=0,
            yes=True,
            classifier=gemini_bad_creds,
            config=config,
        )

    assert mock_post_no_call.call_count == 0
    assert gemini_bad_creds.request_count == 0
    assert res_cred_err["api_calls_made"] == 0
    captured_cred_err = capsys.readouterr().out
    assert "API calls made: 0" in captured_cred_err
