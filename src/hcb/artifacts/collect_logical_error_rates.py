import math
import pathlib
from typing import Iterator, List

from hcb.codes.honeycomb.layout import HoneycombLayout
from hcb.tools.analysis.collecting import collect_simulated_experiment_data, DecodingProblem

OUT_DIR = pathlib.Path(__file__).parent.parent.parent.parent / "out"


def iter_problems(decoders: List[str]) -> Iterator[DecodingProblem]:
    error_rates = [
        0.001,
        0.0015,
        0.002,
        0.003,
        0.005,
        0.007,

        0.03,
        0.02,
        0.015,
        0.01,

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
        3,
        5,
        7,
        9,
        11,
    ]
    for p in error_rates:
        for d in distances:
            for sheared in shearings:
                for decoder in decoders:
                    for obs in observables:
                        for noisy_gate_set in gate_sets:
                            w, h = HoneycombLayout.unsheared_size_for_code_distance(
                                distance=d,
                                gate_set=noisy_gate_set,
                            )
                            yield HoneycombLayout(
                                data_width=w,
                                data_height=h,
                                noise_level=p,
                                noisy_gate_set=noisy_gate_set,
                                tested_observable=obs,
                                sheared=sheared,
                                rounds=math.ceil(d * 3 / 2) * 2,
                            ).to_decoding_problem(decoder=decoder)


def main():
    data_file = OUT_DIR / 'data_fourth.csv'

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


if __name__ == '__main__':
    main()
