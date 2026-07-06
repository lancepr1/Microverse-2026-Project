import history_store


def _init_temp_db(tmp_path):
    path = str(tmp_path / "history_test.db")
    history_store.init_history_db(path)
    return path


def test_record_and_get_recent_round_trip(tmp_path):
    _init_temp_db(tmp_path)

    history_store.record_sample("Rack_01", {
        "index": 0, "frq_hz": 59.5, "total_power_w": 160.0,
        "average_gpu_temp_c": 40.0, "status": "--",
    })
    history_store.record_sample("Rack_01", {
        "index": 1, "frq_hz": 59.6, "total_power_w": 161.0,
        "average_gpu_temp_c": 41.0, "status": "--",
    })

    rows = history_store.get_recent("Rack_01")
    assert len(rows) == 2
    # most recent first
    assert rows[0]["sample_index"] == 1
    assert rows[0]["frq_hz"] == 59.6
    assert rows[1]["sample_index"] == 0


def test_get_recent_respects_limit(tmp_path):
    _init_temp_db(tmp_path)

    for i in range(5):
        history_store.record_sample("Rack_01", {
            "index": i, "frq_hz": 60.0, "total_power_w": 100.0,
            "average_gpu_temp_c": 30.0, "status": "--",
        })

    rows = history_store.get_recent("Rack_01", limit=2)
    assert len(rows) == 2
    assert rows[0]["sample_index"] == 4


def test_get_recent_filters_by_rack_id(tmp_path):
    _init_temp_db(tmp_path)

    history_store.record_sample("Rack_01", {"index": 0, "frq_hz": 60.0})
    history_store.record_sample("Rack_02", {"index": 0, "frq_hz": 60.0})

    rows = history_store.get_recent("Rack_01")
    assert len(rows) == 1
