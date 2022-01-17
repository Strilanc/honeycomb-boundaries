import pathlib
from typing import Iterator

from hcb.codes import generate_surface_code_memory_problem
from hcb.codes.honeycomb.layout import HoneycombLayout
from hcb.tools.analysis.collecting import collect_simulated_experiment_data, read_recorded_data, DecodingProblem, ShotData
from hcb.tools.analysis.plotting import plot_data


def iter_problems(decoder: str) -> Iterator[DecodingProblem]:
    for p in [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05][::-1]:
        for d in [3, 5, 7]:
            yield generate_surface_code_memory_problem(
                noise=p,
                distance=d,
                rounds=d * 3,
                basis="transversal_Z",
                decoder=decoder,
            ).decoding_problem

        for d in [4, 6, 8]:
            yield HoneycombLayout.from_code_distance(
                noise=p,
                distance=d,
                rounds=d * 3,
            ).to_decoding_problem(decoder=decoder)


def main():
    data_file = pathlib.Path("../out/data.csv").resolve()

    collect_simulated_experiment_data(
        iter_problems(decoder="internal"),
        out_path=data_file,
        merge_mode="saturate",
        start_batch_size=2**8,
        max_batch_size=2**18,
        max_shots=100_000,
        max_errors=1000,
        num_threads=8,
    )

    plot_data(read_recorded_data(data_file),
              show=True,
              title="Memory",
              x_max=1,
              y_max=1)


if __name__ == '__main__':
    main()
