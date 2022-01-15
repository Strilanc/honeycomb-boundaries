import stim

from .decoding import sample_decode_count_correct


def test_pymatching_runs():
    num_correct = sample_decode_count_correct(
        num_shots=1000,
        circuit=stim.Circuit.generated(
            "surface_code:rotated_memory_x",
            distance=3,
            rounds=15,
            after_clifford_depolarization=0.001,
        ),
        decoder="pymatching",
    )
    assert 950 <= num_correct <= 1000
