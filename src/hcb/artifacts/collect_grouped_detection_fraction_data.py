import pathlib
import time
from typing import Optional, Union, Iterable, Iterator
import numpy as np

from hcb.codes.honeycomb.layout import HoneycombLayout
from hcb.tools.analysis.collecting import DecodingProblem, CSV_HEADER, \
    CSV_HEADER_VERSION

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
        'EM3_v2',
        'SD6',
        'SI1000',
    ]
    for p in error_rates:
        for noisy_gate_set in gate_sets:
            d = 12
            yield HoneycombLayout(
                data_width=d,
                data_height=(d * 2 + 2) // 3 * 3,
                noise_level=p,
                noisy_gate_set=noisy_gate_set,
                tested_observable='H',
                sheared=False,
                rounds=d * 3,
            ).to_decoding_problem(decoder=decoder)


def collect_grouped_detection_fraction_data(
        problems: Iterable[DecodingProblem],
        *,
        shots: int,
        out_path: Optional[Union[str, pathlib.Path]],
        discard_previous_data: bool):
    print(CSV_HEADER, flush=True)
    if out_path is not None:
        if discard_previous_data or not pathlib.Path(out_path).exists():
            with open(out_path, "w") as f:
                print(CSV_HEADER, file=f)

    for problem in problems:
        circuit = problem.circuit_maker()
        detector_coords = circuit.get_detector_coordinates()
        ndets = circuit.num_detectors
        masks = [
            np.array([detector_coords[k][2] % 6 == r for k in range(ndets)])
            for r in range(6)
        ]

        t0 = time.monotonic()
        samples = problem.circuit_maker().compile_detector_sampler().sample(shots)
        t1 = time.monotonic()
        for r in range(6):
            m = masks[r]
            num_detections = np.count_nonzero(samples & m)
            num_samples = samples.shape[0] * np.count_nonzero(m)
            record = ",".join(str(e) for e in [
                problem.desc.data_width,
                problem.desc.data_height,
                problem.desc.rounds,
                problem.desc.noise,
                problem.desc.circuit_style,
                "-",
                problem.desc.code_distance,
                problem.desc.num_qubits,
                num_samples,
                num_samples - num_detections,
                t1 - t0,
                f"detection_fraction_{r}",
                CSV_HEADER_VERSION,
            ])
            if out_path is not None:
                with open(out_path, "a") as f:
                    print(record, file=f)
            print(record, flush=True)


def main():
    data_file = OUT_DIR / "bounded_honeycomb_dets_grouped.csv"
    collect_grouped_detection_fraction_data(
        problems=iter_problems(decoder="internal"),
        shots=2**10,
        out_path=data_file,
        discard_previous_data=True,
    )


if __name__ == '__main__':
    main()
