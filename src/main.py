import pathlib
from typing import Iterator

from hcb.codes import generate_surface_code_memory_problem
from hcb.tools.analysis.collecting import collect_simulated_experiment_data, read_recorded_data, DecodingProblem, ShotData
from hcb.tools.analysis.plotting import plot_data


def iter_problems(decoder: str) -> Iterator[DecodingProblem]:
    for p in [0.008, 0.005, 0.002, 0.001, 5e-4, 2e-4]:
        for d in [3, 5, 7, 9, 11]:
            yield generate_surface_code_memory_problem(
                noise=p,
                distance=d,
                rounds=d * 3,
                basis="transversal_Z",
                decoder=decoder,
            ).decoding_problem


def main():
    data_file = pathlib.Path("../out/data.csv").resolve()

    collect_simulated_experiment_data(
        iter_problems(decoder="internal"),
        out_path=data_file,
        merge_mode="saturate",
        start_batch_size=100,
        max_shots=10**6,
        max_errors=1000,
        num_threads=8,
    )

    plot_data(read_recorded_data(data_file),
              show=True,
              title="Memory")


if __name__ == '__main__':
    main()
