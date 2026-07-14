import data_feed


def _write_run_file(tmp_path, lines):
    path = tmp_path / "run_test.jsonl"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_init_feed_loads_default_run01(monkeypatch):
    """Smoke test: the real data/run01.jsonl parses end-to-end without error."""
    data_feed.init_feed()
    assert data_feed._ready[0] is True
    assert len(data_feed._samples) == 1800


def test_poll_returns_first_sample_immediately(tmp_path, monkeypatch):
    _write_run_file(tmp_path, [
        '{"index": 0, "FRQ": 59.5, "gpu-0[W]": 70.0, "gpu-0[C]": 40.0, "cpu-0[W]": 90.0, "cpu-0[uJ]": 1000.0}',
        '{"index": 1, "FRQ": 59.6, "gpu-0[W]": 71.0, "gpu-0[C]": 41.0, "cpu-0[W]": 91.0, "cpu-0[uJ]": 1010.0}',
    ])
    monkeypatch.setattr(data_feed, "DATA_DIR", str(tmp_path))
    data_feed.init_feed("run_test.jsonl")

    state = data_feed.poll()
    assert data_feed.get_rack_id() in state
    rack = state[data_feed.get_rack_id()]
    assert rack["frq_hz"] == 59.5
    assert rack["total_power_w"] == 70.0 + 90.0
    assert rack["average_gpu_temp_c"] == 40.0


def test_poll_returns_empty_before_next_replay_interval(tmp_path, monkeypatch):
    _write_run_file(tmp_path, [
        '{"index": 0, "FRQ": 59.5}',
        '{"index": 1, "FRQ": 59.6}',
    ])
    monkeypatch.setattr(data_feed, "DATA_DIR", str(tmp_path))
    data_feed.init_feed("run_test.jsonl")

    data_feed.poll()
    state = data_feed.poll()
    assert state == {}


def test_poll_returns_empty_after_exhausting_samples(tmp_path, monkeypatch):
    _write_run_file(tmp_path, ['{"index": 0, "FRQ": 59.5}'])
    monkeypatch.setattr(data_feed, "DATA_DIR", str(tmp_path))
    data_feed.init_feed("run_test.jsonl")

    data_feed.poll()
    data_feed._cursor[0] = len(data_feed._samples)
    assert data_feed.poll() == {}
