"""
Collects Monte Carlo samples.

Example usage:

PYTHONPATH=src python src/hcb/artifacts/eta.py \
    -case_error_rates 0.03 0.02 0.015 0.01 0.007 0.005 0.003 0.002 0.0015 0.001 0.0007 0.0005 0.0003 0.0002 0.0001 \
    -case_observables H V \
    -case_gate_sets SD6 SI1000 EM3_v2 \
    -case_decoders internal internal_correlated \
    -case_distances 3 5 7 9 11 \
    -max_shots 100_000_000 \
    -max_errors 1000 \
    -stored tmp.csv

"""

import argparse
import math

from hcb.artifacts.collect_logical_error_rates import iter_problems
from hcb.tools.analysis.collecting import MultiStats, CaseStats


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
    parser.add_argument('-stored',
                        type=str,
                        nargs='+',
                        help='Data files to read results from.')

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

    data = MultiStats.from_recorded_data(*args.stored)
    remaining_time = 0
    unknowns = 0
    for p in problems:
        stats = data.data.get(p.desc, CaseStats())
        shots_left = max(args.max_shots - stats.num_shots, 0)
        errors_left = max(args.max_errors - stats.num_errors, 0)
        if stats.num_errors > 5:
            estimated_shots_to_get_errors = math.ceil(errors_left / stats.logical_error_rate * 2)
            shots_left = min(shots_left, estimated_shots_to_get_errors)
        if stats.num_shots > 5:
            dt = stats.total_processing_seconds / stats.num_shots * shots_left
            estimated_time = math.ceil(dt / 60 / 60 * 10) / 10
            remaining_time += dt
        else:
            estimated_time = '???'
            unknowns += 1
        if shots_left:
            print(f'n={shots_left}'.ljust(20) + f't={estimated_time}h'.ljust(20) + f'case={stats.to_csv_line(p.desc)}')
    if remaining_time == 0 and unknowns == 0:
        print("No work left to do")
    else:
        print(f'ETA {math.ceil(remaining_time / 60 / 60 * 10) / 10} core hours (with {unknowns} ??? cases not included)')


if __name__ == '__main__':
    main()
