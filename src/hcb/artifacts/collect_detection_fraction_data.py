import pathlib
from typing import Iterator

from hcb.codes.honeycomb.layout import HoneycombLayout
from hcb.tools.analysis.collecting import DecodingProblem, collect_detection_fraction_data

OUT_DIR = pathlib.Path(__file__).parent.parent.parent.parent / "out"


def iter_problems(decoder: str) -> Iterator[DecodingProblem]:
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
        'EM3_v1',
        'EM3_v2',
        'SD6',
        'SI1000',
    ]
    shearings = [
        False,
        True,
    ]
    for p in error_rates:
        for noisy_gate_set in gate_sets:
            for d in [4, 8, 12, 16, 20]:
                for sheared in shearings:
                    yield HoneycombLayout(
                        data_width=d,
                        data_height=(d * 2 + 2) // 3 * 3,
                        noise_level=p,
                        noisy_gate_set=noisy_gate_set,
                        tested_observable='H',
                        sheared=sheared,
                        rounds=d * 3,
                    ).to_decoding_problem(decoder=decoder)


def main():
    data_file = OUT_DIR / "bounded_honeycomb_dets.csv"
    collect_detection_fraction_data(
        problems=iter_problems(decoder="internal"),
        shots=2**10,
        out_path=data_file,
        discard_previous_data=True,
    )


if __name__ == '__main__':
    main()
