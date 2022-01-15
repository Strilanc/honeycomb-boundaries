import pytest
import stim
import numpy as np

from .noise import NoiseModel, mix_probability_to_independent_component_probability


def test_StandardDepolarizing():
    assert NoiseModel.StandardDepolarizing(0.125).noisy_circuit(stim.Circuit("""
    """)) == stim.Circuit("""
    """)

    assert NoiseModel.StandardDepolarizing(0.125).noisy_circuit(stim.Circuit("""
        CX 1 2
    """)) == stim.Circuit("""
        CX 1 2
        DEPOLARIZE2(0.125) 1 2
        DEPOLARIZE1(0.125) 0
    """)

    assert NoiseModel.StandardDepolarizing(0.125).noisy_circuit(stim.Circuit("""
        CX 1 2
        TICK
    """)) == stim.Circuit("""
        CX 1 2
        DEPOLARIZE2(0.125) 1 2
        DEPOLARIZE1(0.125) 0
        TICK
    """)

    assert NoiseModel.StandardDepolarizing(0.125).noisy_circuit(stim.Circuit("""
        CX 1 2
        TICK
        H 2
    """)) == stim.Circuit("""
        CX 1 2
        DEPOLARIZE2(0.125) 1 2
        DEPOLARIZE1(0.125) 0
        TICK
        H 2
        DEPOLARIZE1(0.125) 2 0 1
    """)

    assert NoiseModel.StandardDepolarizing(0.125).noisy_circuit(stim.Circuit("""
        M 1
    """)) == stim.Circuit("""
        X_ERROR(0.125) 1
        M 1
        DEPOLARIZE1(0.125) 0
    """)

    assert NoiseModel.StandardDepolarizing(0.125).noisy_circuit(stim.Circuit("""
        R 1
    """)) == stim.Circuit("""
        R 1
        X_ERROR(0.125) 1
        DEPOLARIZE1(0.125) 0
    """)

    assert NoiseModel.StandardDepolarizing(0.125).noisy_circuit(stim.Circuit("""
        R 2
        TICK
        REPEAT 100 {
            CX 0 1
            TICK
        }
    """)) == stim.Circuit("""
        R 2
        X_ERROR(0.125) 2
        DEPOLARIZE1(0.125) 0 1
        TICK
        REPEAT 100 {
            CX 0 1
            DEPOLARIZE2(0.125) 0 1
            DEPOLARIZE1(0.125) 2
            TICK
        }
    """)
    with pytest.raises(NotImplementedError):
        NoiseModel.StandardDepolarizing(0.125).noisy_circuit(stim.Circuit("""
            MPP X1*X2
        """))


def test_mix_probability_to_independent_component_probability():

    def independent_samples_distribution(p: float, n: int) -> np.ndarray:
        dist = np.zeros(2**n, dtype=np.float64)
        dist[0] = 1
        for k in range(1, 2**n):
            for src in range(2**n):
                dst = src ^ k
                if src < dst:
                    a, b = dist[src], dist[dst]
                    a2 = a * (1 - p) + b * p
                    b2 = b * (1 - p) + a * p
                    dist[src], dist[dst] = a2, b2
        return dist

    p = 0.1
    actual_dist = independent_samples_distribution(mix_probability_to_independent_component_probability(p, 5), 5)
    expected_dist = [1 - p + p/32] + [p/32]*31
    np.testing.assert_allclose(actual_dist, expected_dist)
