from models import TelemetrySample

SAMPLE_DICT = {
    "index": 0,
    "FRQ": 58.679320001633705,
    "gpu-0[W]": 71.1664, "gpu-1[W]": 73.8635,
    "gpu-0[C]": 40.0,    "gpu-1[C]": 39.0,
    "cpu-0[uJ]": 53150485700.7, "cpu-0-core[uJ]": 11294377312.05,
    "cpu-0[W]": 97.0162,        "cpu-0-core[W]": 0.2087,
}


def test_from_dict_parses_frq_and_index():
    sample = TelemetrySample.from_dict(SAMPLE_DICT)
    assert sample.index == 0
    assert sample.frq_hz == 58.679320001633705


def test_from_dict_parses_gpu_fields_by_index():
    sample = TelemetrySample.from_dict(SAMPLE_DICT)
    assert sample.gpu_power_w == {0: 71.1664, 1: 73.8635}
    assert sample.gpu_temp_c == {0: 40.0, 1: 39.0}


def test_from_dict_parses_cpu_package_vs_core_fields():
    sample = TelemetrySample.from_dict(SAMPLE_DICT)
    assert sample.cpu_power_w == {0: 97.0162}
    assert sample.cpu_energy_uj == {0: 53150485700.7}
    assert sample.cpu_core_power_w == {0: 0.2087}
    assert sample.cpu_core_energy_uj == {0: 11294377312.05}


def test_total_and_average_aggregates():
    sample = TelemetrySample.from_dict(SAMPLE_DICT)
    assert sample.total_gpu_power_w == 71.1664 + 73.8635
    assert sample.total_cpu_power_w == 97.0162
    assert sample.total_power_w == sample.total_gpu_power_w + sample.total_cpu_power_w
    assert sample.average_gpu_temp_c == 39.5


def test_average_gpu_temp_c_none_when_no_gpus():
    sample = TelemetrySample.from_dict({"index": 0, "FRQ": 60.0})
    assert sample.average_gpu_temp_c is None
    assert sample.gpu_power_w == {}


def test_from_json_line():
    import json
    line = json.dumps(SAMPLE_DICT)
    sample = TelemetrySample.from_json_line(line)
    assert sample.frq_hz == SAMPLE_DICT["FRQ"]
