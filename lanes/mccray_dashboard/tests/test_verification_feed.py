import json

import verification_feed


def _write_verification_file(runs_dir, run_id, records):
    run_dir = runs_dir / run_id
    run_dir.mkdir()
    path = run_dir / "verification.jsonl"
    lines = [json.dumps({"_type": "VerificationResult", "data": data}) for data in records]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_verify_sample_reports_worst_status_across_channels(tmp_path, monkeypatch):
    monkeypatch.setattr(verification_feed, "_RUNS_DIR", str(tmp_path))
    _write_verification_file(tmp_path, "run_test", [
        {"timestamp": 0.0, "component_id": "run_test/ENF", "status": "trusted",
         "score": 0.95, "anchor_ref": 0.0, "reason": "ok"},
        {"timestamp": 0.0, "component_id": "run_test/gpu-0[W]", "status": "failed",
         "score": 0.05, "anchor_ref": 0.0,
         "reason": "OUT OF RANGE: gpu-0[W]=-5.00W is negative"},
    ])

    verification_feed.init_verifier("run_test")
    result = verification_feed.verify_sample(0)

    assert result["status"] == "warning"
    assert result["reasons"] == [("gpu-0[W]", "OUT OF RANGE: gpu-0[W]=-5.00W is negative")]


def test_verify_sample_all_trusted_reports_good(tmp_path, monkeypatch):
    monkeypatch.setattr(verification_feed, "_RUNS_DIR", str(tmp_path))
    _write_verification_file(tmp_path, "run_test", [
        {"timestamp": 0.0, "component_id": "run_test/ENF", "status": "trusted",
         "score": 0.95, "anchor_ref": 0.0, "reason": "ok"},
    ])

    verification_feed.init_verifier("run_test")
    result = verification_feed.verify_sample(0)

    assert result["status"] == "good"
    assert result["reasons"] == []


def test_verify_sample_missing_run_falls_back_to_placeholder(tmp_path, monkeypatch):
    monkeypatch.setattr(verification_feed, "_RUNS_DIR", str(tmp_path))

    verification_feed.init_verifier("no_such_run")
    result = verification_feed.verify_sample(0)

    assert result["status"] == "--"
    assert result["reasons"] == []
