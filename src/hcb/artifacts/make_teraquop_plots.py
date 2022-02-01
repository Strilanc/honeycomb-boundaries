import math
import pathlib
import sys
from typing import List, Tuple, Dict
import numpy as np

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from scipy.stats import linregress

from hcb.artifacts.make_lambda_plots import DesiredLineFit, project_intersection_of_both_observables
from hcb.tools.analysis.collecting import read_recorded_data, MultiStats, DecodingProblemDesc
from hcb.tools.analysis.probability_util import least_squares_output_range

OUT_DIR = pathlib.Path("../../../out/").resolve()

MARKERS = "ov*sp^<>8PhH+xXDd|" * 100
COLORS = list(mcolors.TABLEAU_COLORS) * 3


def main():
    if len(sys.argv) == 1:
        raise ValueError("Specify csv files to include as command line arguments.")

    csvs = []
    for path in sys.argv[1:]:
        p = pathlib.Path(path)
        if p.is_dir():
            csvs.extend(p.glob("*.csv"))
        else:
            csvs.append(p)

    all_data = read_recorded_data(*csvs).filter(lambda e: 'PC3' not in e.circuit_style)

    gate_sets = {
        'SD6': 'SD6',
        'SI1000': 'SI1000',
        'EM3': 'EM3_v2',
    }
    layouts = {
        'toric (standard)': ('honeycomb', 'internal'),
        'toric (correlated)': ('honeycomb', 'internal_correlated'),
        'bounded (standard)': ('bounded_honeycomb_memory', 'internal'),
        'bounded (correlated)': ('bounded_honeycomb_memory', 'internal_correlated'),
    }
    groups = {
        gate_set_caption: [
            DesiredLineFit(
                legend_caption=layout_caption,
                marker=MARKERS[k],
                color=COLORS[k],
                filter_circuit_style=f"{gate_set_prefix}_{gate_set_suffix}",
                filter_decoder=decoder,
            )
            for k, (layout_caption, (gate_set_prefix, decoder)) in enumerate(layouts.items())
        ]
        for gate_set_caption, gate_set_suffix in gate_sets.items()
    }

    fig2, _ = make_teraquop_plots(all_data, groups)
    fig2.set_size_inches(13, 7)
    fig2.savefig(OUT_DIR / "teraquop.png", bbox_inches='tight', dpi=200)

    plt.show()


def make_teraquop_plots(
        data: MultiStats,
        groups: Dict[str, List[DesiredLineFit]]):
    data = project_intersection_of_both_observables(data)
    grouped = data.grouped_by(lambda e: (e.circuit_style, e.decoder))

    fig, axs = plt.subplots(1, len(groups))
    for k, (name, gs) in enumerate(groups.items()):
        fill_in_single_teraquop_plot(grouped, gs, axs[k])
        axs[k].set_title(name)
    axs[-1].legend()

    return fig, axs


def extrapolate_qubit_count(*,
                            starts: List[DecodingProblemDesc],
                            new_distance: float) -> int:
    style, = {e.circuit_style for e in starts}
    if style.startswith("bounded_honeycomb_memory_"):
        quant = 2
    elif style.startswith("honeycomb_"):
        quant = 4
    else:
        raise NotImplementedError()
    a, b, c = np.polyfit([e.data_width for e in starts], [e.num_qubits for e in starts], 2)
    d = math.ceil(new_distance / quant) * quant
    return d*d*a + d*b + c


def teraquop_intercept_range(*,
                             starts: List[DecodingProblemDesc],
                             xs: List[float],
                             ys: List[float]) -> Tuple[float, float, float]:
    assert len(xs) > 1
    assert len(xs) == len(ys)
    dlow, dmid, dhigh = least_squares_output_range(
        xs=ys,
        ys=xs,
        target_x=math.log(1e-12),
        cost_increase=1)
    qlow, qmid, qhigh = [
        extrapolate_qubit_count(starts=starts, new_distance=d)
        for d in (dlow, dmid, dhigh)
    ]
    return qlow, qmid, qhigh


def fill_in_single_teraquop_plot(
    grouped_projected_data: Dict[Tuple[str, str], MultiStats],
    groups: List[DesiredLineFit],
    ax: plt.Axes,
):
    for group in groups:
        stats = grouped_projected_data.get((group.filter_circuit_style, group.filter_decoder), MultiStats({}))
        xs = []
        ys = []
        ys_low = []
        ys_high = []
        for noise, noise_stats in stats.grouped_by(lambda e: e.noise).items():
            linefit_xs, linefit_ys = noise_stats.after_discarding_degenerates().to_xs_ys(
                x_func=lambda e: e.data_width,
                y_func=lambda e: math.log(e.logical_error_rate),
            )
            if len(linefit_xs) < 2:
                continue
            if linregress(linefit_xs, linefit_ys).slope > -0.001:
                continue

            low, best, high = teraquop_intercept_range(
                starts=list(noise_stats.data.keys()),
                xs=linefit_xs,
                ys=linefit_ys)
            xs.append(noise)
            ys_low.append(low)
            ys.append(best)
            ys_high.append(high)
        ax.plot(xs, ys, marker=group.marker, label=group.legend_caption, color=group.color, zorder=10)
        ax.fill_between(xs, ys_low, ys_high, alpha=0.1, color=group.color)

        ax.set_xlabel("Physical Error Rate")
        ax.set_ylabel("Teraquop Qubit Count")
        ax.set_xlim(1e-4, 2e-2)
        ax.set_ylim(100, 100_000)
        ax.loglog()
        ax.grid(which='minor', color='#AAAAAA')
        ax.grid(which='major', color='black')


if __name__ == '__main__':
    main()
