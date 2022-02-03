from hcb.artifacts.make_teraquop_plots import extrapolate_num_qubits
from hcb.codes.honeycomb.layout import HoneycombLayout
from hcb.tools.analysis.collecting import DecodingProblemDesc


def test_extrapolate_num_qubits_planar():
    w, h = HoneycombLayout.unsheared_size_for_code_distance(distance=5, gate_set='SD6')
    layout = HoneycombLayout(
        data_width=w,
        data_height=h,
        rounds=10,
        noise_level=0.001,
        noisy_gate_set='SD6',
        tested_observable='H',
        sheared=False,
    )
    for d in [5, 6, 7, 13, 14, 15]:
        w, h = HoneycombLayout.unsheared_size_for_code_distance(distance=d, gate_set='SD6')
        expected_layout = HoneycombLayout(
            data_width=w,
            data_height=h,
            rounds=10,
            noise_level=0.001,
            noisy_gate_set='SD6',
            tested_observable='H',
            sheared=False,
        )
        assert extrapolate_num_qubits(
            bases=[layout.to_decoding_desc(decoder='???')],
            new_code_distance=d,
        ) == expected_layout.num_used_qubits()


def test_extrapolate_num_qubits_periodic_sd6():
    w, h = HoneycombLayout.unsheared_size_for_code_distance(distance=5, gate_set='SD6')
    layout = DecodingProblemDesc(
        data_width=w,
        data_height=h,
        rounds=10,
        code_distance=5,
        num_qubits=5*5*60,
        noise=0,
        circuit_style='honeycomb_SD6',
        preserved_observable='H',
        decoder='',
    )
    expected = [
        None,
        60,
        60,
        60,
        60,
        240,
        240,
        240,
        240,
        540,
        540,
        540,
        540,
    ]
    for k in range(len(expected)):
        if expected[k] is not None:
            assert extrapolate_num_qubits(bases=[layout], new_code_distance=k) == expected[k]


def test_extrapolate_num_qubits_periodic_em3():
    w, h = HoneycombLayout.unsheared_size_for_code_distance(distance=5, gate_set='SD6')
    layout = DecodingProblemDesc(
        data_width=w,
        data_height=h,
        rounds=10,
        code_distance=5,
        num_qubits=5*5*60,
        noise=0,
        circuit_style='honeycomb_EM3_v2',
        preserved_observable='H',
        decoder='',
    )
    expected = [
        None,
        24,
        24,
        24,
        24,
        96,
        96,
        96,
        96,
        216,
        216,
        216,
        216,
    ]
    for k in range(len(expected)):
        if expected[k] is not None:
            assert extrapolate_num_qubits(bases=[layout], new_code_distance=k) == expected[k]
