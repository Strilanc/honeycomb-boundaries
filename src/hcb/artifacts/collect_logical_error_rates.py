"""
Collects Monte Carlo samples.

Example usage:

PYTHONPATH=src python src/hcb/artifacts/collect_logical_error_rates.py \
    -case_error_rates 0.03 0.02 0.015 0.01 0.007 0.005 0.003 0.002 0.0015 0.001 0.0007 0.0005 0.0003 0.0002 0.0001 \
    -case_observables H V \
    -case_gate_sets SD6 SI1000 EM3_v2 \
    -case_decoders internal internal_correlated \
    -case_distances 3 5 7 9 11 \
    -max_shots 100_000_000 \
    -max_errors 1000 \
    -merge_mode saturate \
    -storage_location tmp.csv \
    -threads 8

"""

import argparse
import math
from typing import Iterator, List, Sequence

from hcb.codes.honeycomb.layout import HoneycombLayout
from hcb.tools.analysis.collecting import collect_simulated_experiment_data, DecodingProblem


def iter_problems(*,
                  decoders: List[str],
                  error_rates: Sequence[float],
                  shearings: Sequence[bool],
                  gate_sets: Sequence[str],
                  distances: Sequence[int],
                  observables: Sequence[str]) -> Iterator[DecodingProblem]:
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
    parser = argparse.ArgumentParser(description='Collect Monte Carlo samples.')
    parser.add_argument('-max_shots',
                        type=int,
                        required=True,
                        help='Stops sampling a case if this many shots are acquired.')
    parser.add_argument('-max_errors',
                        type=int,
                        required=True,
                        help='Stops sampling a case if this many errors have been seen.')
    parser.add_argument('-start_batch_size',
                        type=int,
                        default=10**2,
                        help='Initial number of samples to batch together into one decoding call (this will be increased exponentially each call until hitting max_batch_size).')
    parser.add_argument('-max_batch_size',
                        type=int,
                        default=10**5,
                        help='Maximum number of samples to batch together into one decoding call.')
    parser.add_argument('-storage_location',
                        type=str,
                        default='',
                        help='The file to store results in (or read previous results from).')
    parser.add_argument('-read_locations',
                        type=str,
                        nargs='+',
                        help='CSV files to include in stats decisions to continue, but not to write to.')
    parser.add_argument('-merge_mode',
                        choices=['saturate', 'replace', 'append'],
                        default='saturate',
                        help='How to combine new results with old results.')
    parser.add_argument('-threads',
                        type=int,
                        default=1,
                        help='Number of threads to use.')

    parser.add_argument('-case_error_rates',
                        type=float,
                        nargs='+',
                        required=True,
                        help='Physical error rates to sample from.')
    parser.add_argument('-case_observables',
                        choices=['H', 'V', 'EPR'],
                        nargs='+',
                        required=True,
                        help='Which observable to intialize and protect and measure.')
    parser.add_argument('-case_gate_sets',
                        choices=['SD6', 'SI1000', 'EM3_v2'],
                        nargs='+',
                        required=True,
                        help='Noisy gate sets to sample.')
    parser.add_argument('-case_distances',
                        type=int,
                        nargs='+',
                        required=True,
                        help='Code distances to sample.')
    parser.add_argument('-case_decoders',
                        choices=['internal', 'internal_correlated', 'pymatching'],
                        nargs='+',
                        required=True,
                        help='How to combine new results with old results.')
    parser.add_argument('-case_sheared',
                        type=bool,
                        default=False,
                        help='Uses sheared layouts.')

    args = parser.parse_args()
    problems = iter_problems(
        decoders=args.case_decoders,
        error_rates=args.case_error_rates,
        distances=args.case_distances,
        observables=args.case_observables,
        shearings=[args.case_sheared],
        gate_sets=args.case_gate_sets,
    )

    collect_simulated_experiment_data(
        problems,
        out_path=args.storage_location or None,
        alt_in_paths=args.read_locations or (),
        merge_mode=args.merge_mode,
        start_batch_size=args.start_batch_size,
        max_batch_size=args.max_batch_size,
        max_shots=args.max_shots,
        max_errors=args.max_errors,
        num_threads=args.threads,
    )


if __name__ == '__main__':
    main()
