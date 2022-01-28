import pathlib
from typing import Iterator, List

from hcb.codes.honeycomb.layout import HoneycombLayout
from hcb.tools.analysis.collecting import collect_simulated_experiment_data, read_recorded_data, DecodingProblem
from hcb.tools.analysis.plotting import plot_data


def iter_problems(decoders: List[str]) -> Iterator[DecodingProblem]:
    error_rates = [
        0.03,
        0.02,
        0.015,
        0.01,
        0.007,
        0.005,
        0.003,
        0.002,
        0.0015,
        0.001,
        0.0007,
        0.0005,
        0.0003,
        0.0002,
        0.0001
    ]
    gate_sets = [
        'EM3_v2',
        'SD6',
        'SI1000',
    ]
    shearings = [
        False,
        # True,
    ]
    observables = [
        'H',
        'V',
        # 'EPR',
    ]
    distances = [
        4,
        8,
        12,
        # 16,
        # 20,
    ]
    for p in error_rates:
        for noisy_gate_set in gate_sets:
            for d in distances:
                for sheared in shearings:
                    for decoder in decoders:
                        for obs in observables:
                            yield HoneycombLayout(
                                data_width=d,
                                data_height=(d * 2 + 2) // 3 * 3,
                                noise_level=p,
                                noisy_gate_set=noisy_gate_set,
                                tested_observable=obs,
                                sheared=sheared,
                                rounds=d * 3,
                            ).to_decoding_problem(decoder=decoder)


def main():
    data_file = pathlib.Path("../out/data_third.csv").resolve()

    collect_simulated_experiment_data(
        iter_problems(decoders=["internal", "internal_correlated"]),
        out_path=data_file,
        merge_mode="saturate",
        start_batch_size=10**2,
        max_batch_size=10**5,
        max_shots=1_000_000,
        max_errors=1000,
        num_threads=6,
    )

    stats = read_recorded_data(data_file)
    stats = stats.filter(lambda e: 'SD6' in e.circuit_style and 'sheared' not in e.circuit_style and e.data_width > 2)
    plot_data(stats,
              show=True,
              title="Memory",
              x_max=1,
              y_max=1)


if __name__ == '__main__':
    main()
