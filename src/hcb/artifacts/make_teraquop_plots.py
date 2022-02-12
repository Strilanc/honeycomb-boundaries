import math
import pathlib
import sys
from typing import List, Tuple, Dict

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from scipy.stats import linregress

from hcb.artifacts.make_lambda_plots import DesiredLineFit, project_intersection_of_both_observables
from hcb.codes.honeycomb.layout import HoneycombLayout
from hcb.tools.analysis.collecting import MultiStats, DecodingProblemDesc
from hcb.tools.analysis.plotting import total_error_to_per_piece_error
from hcb.tools.analysis.probability_util import least_squares_output_range

OUT_DIR = pathlib.Path(__file__).parent.parent.parent.parent / "out"

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

    all_data = MultiStats.from_recorded_data(*csvs).filter(lambda e: 'PC3' not in e.circuit_style)

    gate_sets = {
        'SD6': 'SD6',
        'SI1000': 'SI1000',
        'EM3': 'EM3_v2',
    }
    layouts = {
        'periodic honeycomb code\n(MWPM decoding)': ('honeycomb', 'internal'),
        'periodic honeycomb code\n(correlated MWPM decoding)': ('honeycomb', 'internal_correlated'),
        'planar honeycomb code\n(MWPM decoding)': ('bounded_honeycomb_memory', 'internal'),
        'planar honeycomb code\n(correlated MWPM decoding)': ('bounded_honeycomb_memory', 'internal_correlated'),
    }
    groups = {
        gate_set_caption: [
            DesiredLineFit(
                legend_caption=layout_caption,
                marker=MARKERS[k],
                color=COLORS[k],
                filter_circuit_style=f"{gate_set_prefix}_{gate_set_suffix}",
                filter_decoder=decoder,
                observable="combo",
            )
            for k, (layout_caption, (gate_set_prefix, decoder)) in enumerate(layouts.items())
        ]
        for gate_set_caption, gate_set_suffix in gate_sets.items()
    }

    fig2, _ = make_teraquop_plots(all_data, groups)
    fig2.set_size_inches(18, 6)
    fig2.savefig(OUT_DIR / "teraquop.png", bbox_inches='tight', dpi=200)

    plt.show()


def make_teraquop_plots(
        data: MultiStats,
        groups: Dict[str, List[DesiredLineFit]]):
    data = project_intersection_of_both_observables(data)
    grouped = data.grouped_by(lambda e: (e.circuit_style, e.decoder))

    axs: List[plt.Axes]
    fig, axs = plt.subplots(1, len(groups) + 1)
    for k, (name, gs) in enumerate(groups.items()):
        fill_in_single_teraquop_plot(grouped, gs, axs[k])
        axs[k].set_title(name)
    for k in range(1, len(groups)):
        axs[k].set_yticklabels([])
    axs[0].set_ylabel("Teraquop Qubit Count")
    axs[0].yaxis.label.set_fontsize(14)

    axs[-1].legend(*axs[-2].get_legend_handles_labels(), loc="upper left")
    axs[-1].get_xaxis().set_ticks([])
    axs[-1].get_yaxis().set_ticks([])
    axs[-1].spines['top'].set_visible(False)
    axs[-1].spines['right'].set_visible(False)
    axs[-1].spines['bottom'].set_visible(False)
    axs[-1].spines['left'].set_visible(False)

    return fig, axs


def teraquop_intercept_range(*,
                             starts: List[DecodingProblemDesc],
                             distances: List[float],
                             log_error_rates: List[float]) -> Tuple[float, float, float]:
    assert len(distances) > 1
    assert len(distances) == len(log_error_rates)
    dlow, dmid, dhigh = least_squares_output_range(
        xs=log_error_rates,
        ys=distances,
        target_x=math.log(1e-12),
        cost_increase=1)
    qlow, qmid, qhigh = [
        extrapolate_num_qubits(bases=starts, new_code_distance=d)
        for d in (dlow, dmid, dhigh)
    ]
    return qlow, qmid, qhigh


def fill_in_single_teraquop_plot(
    grouped_projected_data: Dict[Tuple[str, str], MultiStats],
    groups: List[DesiredLineFit],
    ax: plt.Axes,
):
    pieces = 3
    for group in groups:
        stats = grouped_projected_data.get((group.filter_circuit_style, group.filter_decoder), MultiStats({}))
        xs = []
        ys = []
        ys_low = []
        ys_high = []
        for noise, noise_stats in stats.grouped_by(lambda e: e.noise).items():
            linefit_xs, linefit_ys = noise_stats.after_discarding_degenerates().to_xs_ys(
                x_func=lambda e: e.code_distance,
                y_func=lambda e: math.log(total_error_to_per_piece_error(e.logical_error_rate, pieces=pieces)),
            )
            if len(linefit_xs) < 2:
                continue
            if linregress(linefit_xs, linefit_ys).slope > -0.001:
                continue

            low, best, high = teraquop_intercept_range(
                starts=list(noise_stats.data.keys()),
                distances=linefit_xs,
                log_error_rates=linefit_ys)
            xs.append(noise)
            ys_low.append(low)
            ys.append(best)
            ys_high.append(high)
        ax.plot(xs, ys, marker=group.marker, label=group.legend_caption, color=group.color, zorder=10)
        ax.fill_between(xs, ys_low, ys_high, alpha=0.2, color=group.color)

        ax.set_xlabel("Physical Error Rate")
        ax.set_xlim(1e-4, 2e-2)
        ax.set_ylim(100, 100_000)
        ax.loglog()
        ax.grid(which='minor', color='#AAAAAA')
        ax.grid(which='major', color='black')


def extrapolate_num_qubits(*,
                           bases: List[DecodingProblemDesc],
                           new_code_distance: float):
    basis = bases[0]
    if basis.circuit_style.startswith("bounded_honeycomb_memory_"):
        g = basis.circuit_style.split("bounded_honeycomb_memory_")[-1]
        w, h = HoneycombLayout.unsheared_size_for_code_distance(
            distance=math.ceil(new_code_distance),
            gate_set=g)
        return HoneycombLayout(
            data_width=w,
            data_height=h,
            rounds=10,
            noise_level=0,
            noisy_gate_set=g,
            tested_observable='H',
            sheared=False
        ).num_used_qubits()
    elif basis.circuit_style in ["honeycomb_EM3_v2", "honeycomb_EM3"]:
        d = math.ceil(new_code_distance / 4)
        return d * d * 24
    elif basis.circuit_style in ["honeycomb_SD6", "honeycomb_SI1000"]:
        d = math.ceil(new_code_distance / 4)
        return d * d * 60
    else:
        raise NotImplementedError(f'{basis.circuit_style=}')


if __name__ == '__main__':
    main()
