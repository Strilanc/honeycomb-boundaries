import tempfile

import numpy as np
import pytest
import stim

from hcb.tools.analysis.collecting import (
    collect_simulated_experiment_data, CaseStats, DecodingProblem, MultiStats,
    DecodingProblemDesc,
)
from .plotting import plot_data
from .probability_util import log_binomial


def surface_code_problem(noise: float, d: int) -> DecodingProblem:
    return DecodingProblem(
        circuit_maker=lambda: stim.Circuit.generated(
            "surface_code:rotated_memory_x",
            distance=d,
            rounds=d * 3,
            after_clifford_depolarization=noise,
        ),
        desc=DecodingProblemDesc(
            data_width=d,
            data_height=d,
            code_distance=d,
            num_qubits=2 * d * d - 1,
            rounds=3 * d,
            noise=noise,
            circuit_style=f"surface_code_test",
            preserved_observable="X",
            decoder="pymatching"
        ),
    )


def test_collect_and_plot():
    with tempfile.TemporaryDirectory() as d:
        f = d + "/tmp.csv"
        collect_simulated_experiment_data(
            [
                surface_code_problem(
                    noise=p,
                    d=d,
                )
                for p in [1e-4, 1e-3]
                for d in [3, 5]
            ],
            out_path=f,
            merge_mode="replace",
            start_batch_size=10,
            max_shots=10,
            max_errors=1,
        )

        plot_data(MultiStats.from_recorded_data(f), show=False, out_path=d + "/tmp.png", title="Test")


def test_likely_error_rate_bounds_shrink_towards_half():
    np.testing.assert_allclose(
        CaseStats(num_shots=10 ** 5, num_correct=10 ** 5 / 2).likely_error_rate_bounds(desired_ratio_vs_max_likelihood=1e-3),
        (0.494122, 0.505878),
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        CaseStats(num_shots=10 ** 4, num_correct=10 ** 4 / 2).likely_error_rate_bounds(desired_ratio_vs_max_likelihood=1e-3),
        (0.481422, 0.518578),
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        CaseStats(num_shots=10 ** 4, num_correct=10 ** 4 / 2).likely_error_rate_bounds(desired_ratio_vs_max_likelihood=1e-2),
        (0.48483, 0.51517),
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        CaseStats(num_shots=1000, num_correct=500).likely_error_rate_bounds(desired_ratio_vs_max_likelihood=1e-3),
        (0.44143, 0.55857),
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        CaseStats(num_shots=100, num_correct=50).likely_error_rate_bounds(desired_ratio_vs_max_likelihood=1e-3),
        (0.3204, 0.6796),
        rtol=1e-4,
    )


@pytest.mark.parametrize("n,c,ratio", [
    (100, 50, 1e-1),
    (100, 50, 1e-2),
    (100, 50, 1e-3),
    (1000, 500, 1e-3),
    (10**6, 100, 1e-3),
    (10**6, 100, 1e-2),
])
def test_likely_error_rate_bounds_vs_log_binomial(n: int, c: int, ratio: float):

    a, b = CaseStats(num_shots=n, num_correct=c).likely_error_rate_bounds(desired_ratio_vs_max_likelihood=ratio)

    raw = log_binomial(p=(n - c) / n, n=n, hits=n - c)
    low = log_binomial(p=a, n=n, hits=n - c)
    high = log_binomial(p=b, n=n, hits=n - c)
    np.testing.assert_allclose(
        np.exp(low - raw),
        ratio,
        rtol=1e-2,
    )
    np.testing.assert_allclose(
        np.exp(high - raw),
        ratio,
        rtol=1e-2,
    )
