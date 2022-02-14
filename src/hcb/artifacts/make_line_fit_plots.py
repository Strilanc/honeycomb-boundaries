import math
import pathlib
import sys
from typing import List, Tuple, Dict, Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from scipy.stats import linregress

from hcb.artifacts.make_lambda_plots import DesiredLineFit, project_intersection_of_both_observables
from hcb.tools.analysis.collecting import MultiStats
from hcb.tools.analysis.plotting import total_error_to_per_piece_error

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
        'periodic honeycomb code\n(correlated MWPM)': ('honeycomb', 'internal_correlated'),
        'planar honeycomb code\n(correlated MWPM)': ('bounded_honeycomb_memory', 'internal_correlated'),
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

    fig, _ = make_line_fit_plots(all_data, groups)
    fig.set_size_inches(18, 6)
    fig.savefig(OUT_DIR / "line_fit.png", bbox_inches='tight', dpi=200)

    plt.show()


def make_line_fit_plots(
        data: MultiStats,
        groups: Dict[str, List[DesiredLineFit]]):
    data = project_intersection_of_both_observables(data)
    grouped = data.grouped_by(lambda e: (e.circuit_style, e.decoder))
    noises = sorted({desc.noise for desc in data.data.keys()}, reverse=True)
    noise_style = {
        noise: (COLORS[k], MARKERS[k])
        for k, noise in enumerate(noises)
    }

    fig, axs = plt.subplots(2, len(groups) + 1)
    for k, (name, gs) in enumerate(groups.items()):
        for y, g in enumerate(gs):
            fill_in_line_fit_plot(grouped, g, axs[y, k], noise_style)
            if k == 0:
                axs[y, 0].set_ylabel(f"{g.legend_caption}\ncombo logical error rate")
                axs[y, 0].yaxis.label.set_fontsize(14)
        axs[0, k].set_title(name)
        axs[-1, k].set_xlabel("Patch Width")
        axs[-1, k].xaxis.label.set_fontsize(10)
        axs[0, k].title.set_fontsize(14)
    for k in range(len(groups)):
        axs[0, k].set_xticklabels([])
    for k in range(1, len(groups)):
        axs[0, k].set_yticklabels([])
        axs[1, k].set_yticklabels([])

    axs[0, -1].legend(*axs[0, 0].get_legend_handles_labels(), loc="upper left", title="Physical Error Rate")
    axs[0, -1].get_xaxis().set_ticks([])
    axs[0, -1].get_yaxis().set_ticks([])
    axs[0, -1].spines['top'].set_visible(False)
    axs[0, -1].spines['right'].set_visible(False)
    axs[0, -1].spines['bottom'].set_visible(False)
    axs[0, -1].spines['left'].set_visible(False)
    axs[1, -1].set_visible(False)
    return fig, axs


def fill_in_line_fit_plot(
    grouped_projected_data: Dict[Tuple[str, str], MultiStats],
    group: DesiredLineFit,
    ax: plt.Axes,
    noise_style: Dict[float, Tuple[Any, Any]],
):
    stats = grouped_projected_data.get((group.filter_circuit_style, group.filter_decoder), MultiStats({}))
    pieces = 3
    for noise, noise_stats in stats.grouped_by(lambda e: e.noise, reverse=True).items():
        xs = []
        ys = []
        for x, group_stats in noise_stats.grouped_by(lambda e: e.data_width).items():
            assert len(group_stats.data) == 1
            case_stats = group_stats.merged_total()
            xs.append(x)
            ys.append(total_error_to_per_piece_error(case_stats.logical_error_rate, pieces=pieces))
        x2s = []
        y2s = []
        for x, y in zip(xs, ys):
            if y:
                x2s.append(x)
                y2s.append(math.log(y))
        color, marker = noise_style[noise]
        ax.scatter(xs, ys, marker=marker, label=str(noise), color=color, zorder=10)
        if len(y2s) >= 2:
            fit = linregress(x2s, y2s)
            if fit.slope <= -0.02:
                xb = [1, 25]
                yb = [math.exp(fit.intercept + fit.slope * x) for x in xb]
                ax.plot(xb, yb, linestyle='dashed', color=color, zorder=9)

    ax.set_xlim(1, 25)
    ax.set_ylim(1e-12, 1e-0)
    ax.semilogy()
    ax.grid(which='minor', color='#AAAAAA')
    ax.grid(which='major', color='black')


if __name__ == '__main__':
    main()
