import pytest

from hcb.codes.honeycomb.layout import HoneycombLayout


@pytest.mark.parametrize("code_distance,gate_set,tested_observable", [
    (d, g, o)
    for d in [1, 2, 3, 4, 5, 9, 10, 11]
    for g in ["SI1000", "SD6", "EM3_v2"]
    for o in ['H', 'V', 'EPR']
])
def test_graphlike_code_distances(*,
                                  code_distance: int,
                                  gate_set: str,
                                  tested_observable: str):
    w, h = HoneycombLayout.unsheared_size_for_code_distance(
        distance=code_distance,
        gate_set=gate_set,
    )
    layout = HoneycombLayout(data_width=w,
                             data_height=h,
                             rounds=10,
                             noise_level=0.001,
                             noisy_gate_set=gate_set,
                             tested_observable=tested_observable)
    circuit = layout.noisy_circuit()
    err = circuit.shortest_graphlike_error(ignore_ungraphlike_errors=True)
    allowed = [code_distance]
    if tested_observable == 'H':
        allowed.append(code_distance + 1)
    assert len(err) in allowed
    assert layout.horizontal_graphlike_code_distance() == code_distance
    assert layout.vertical_graphlike_code_distance() in [code_distance, code_distance + 1]
    assert layout.graphlike_code_distance() == code_distance


@pytest.mark.parametrize("w,h,g", [
    (w, h, g)
    for w in [2, 3, 4, 5, 6]
    for h in [3, 6, 9, 12, 15]
    for g in ["SD6", "EM3_v2"]
])
def test_num_data_qubits(w: int, h: int, g: str):
    layout = HoneycombLayout(data_width=w,
                             data_height=h,
                             rounds=10,
                             noise_level=0,
                             noisy_gate_set=g,
                             tested_observable='H')
    assert layout.num_data_qubits() == len(layout.data_qubit_set)


@pytest.mark.parametrize("w,h,g", [
    (w, h, g)
    for w in [2, 3, 4, 5, 6, 7, 8, 9]
    for h in [3, 6, 9, 12, 15, 18, 21]
    for g in ["SD6", "EM3_v2"]
])
def test_num_used_qubits(w: int, h: int, g: str):
    layout = HoneycombLayout(data_width=w,
                             data_height=h,
                             rounds=10,
                             noise_level=0,
                             noisy_gate_set=g,
                             tested_observable='H')
    if g == "EM3_v2":
        assert layout.num_used_qubits() == len(layout.data_qubit_set)
    else:
        assert layout.num_used_qubits() == len(layout.all_qubits_set)
