import pathlib
import sys
from typing import List, Tuple, Dict, Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from hcb.artifacts.make_lambda_plots import DesiredLineFit, project_intersection_of_both_observables
from hcb.tools.analysis.collecting import read_recorded_data, MultiStats

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

    all_data = read_recorded_data(*csvs).filter(lambda e: 'PC3' not in e.circuit_style)

    gate_sets = {
        'SD6': 'SD6',
        'SI1000': 'SI1000',
        'EM3': 'EM3_v2',
    }
    layouts = {
        'toric (correlated)': ('honeycomb', 'internal_correlated'),
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

    fig, _ = make_threshold_plots(all_data, groups)
    fig.set_size_inches(13, 7)
    fig.savefig(OUT_DIR / "threshold.png", bbox_inches='tight', dpi=200)

    plt.show()


def make_threshold_plots(
        data: MultiStats,
        groups: Dict[str, List[DesiredLineFit]]):
    data = project_intersection_of_both_observables(data)
    grouped = data.grouped_by(lambda e: (e.circuit_style, e.decoder))
    widths = sorted({desc.data_width for desc in data.data.keys()}, reverse=True)
    width_style = {
        noise: (COLORS[k], MARKERS[k])
        for k, noise in enumerate(widths)
    }

    fig, axs = plt.subplots(2, len(groups) + 1)
    for k, (name, gs) in enumerate(groups.items()):
        for y, g in enumerate(gs):
            fill_in_threshold_plot(grouped, g, axs[y, k], width_style)
            if k == 0:
                axs[y, 0].set_ylabel(f"{g.legend_caption}\nLogical Error Rate")
                axs[y, 0].yaxis.label.set_fontsize(14)
        axs[0, k].set_title(name)
        axs[-1, k].set_xlabel("Physical Error Rate")
        axs[-1, k].xaxis.label.set_fontsize(14)

    for y in [0, 1]:
        axs[y, -1].legend(*axs[y, 0].get_legend_handles_labels(), loc="upper left", title="Size")
        axs[y, -1].get_xaxis().set_ticks([])
        axs[y, -1].get_yaxis().set_ticks([])
        axs[y, -1].spines['top'].set_visible(False)
        axs[y, -1].spines['right'].set_visible(False)
        axs[y, -1].spines['bottom'].set_visible(False)
        axs[y, -1].spines['left'].set_visible(False)
    return fig, axs


def fill_in_threshold_plot(
    grouped_projected_data: Dict[Tuple[str, str], MultiStats],
    group: DesiredLineFit,
    ax: plt.Axes,
    width_style: Dict[float, Tuple[Any, Any]],
):
    stats = grouped_projected_data.get((group.filter_circuit_style, group.filter_decoder), MultiStats({}))
    for data_width, noise_stats in stats.grouped_by(lambda e: e.data_width).items():
        xs = []
        ys = []
        ys_low = []
        ys_high = []
        for noise, group_stats in noise_stats.grouped_by(lambda e: e.noise).items():
            assert len(group_stats.data) == 1
            case_stats = group_stats.merged_total()
            xs.append(noise)
            ys.append(case_stats.logical_error_rate)
            low, high = case_stats.likely_error_rate_bounds(desired_ratio_vs_max_likelihood=1e-3)
            ys_low.append(low)
            ys_high.append(high)
        color, marker = width_style[data_width]
        rep = next(iter(noise_stats.data))
        ax.plot(xs, ys, marker=marker, label=f'{rep.data_width}x{rep.data_height} rounds={rep.rounds}', color=color, zorder=10)
        ax.fill_between(xs, ys_low, ys_high, alpha=0.3, color=color)

    ax.set_xlim(1e-4, 1e-1)
    ax.set_ylim(1e-6, 1e-0)
    ax.loglog()
    ax.grid(which='minor', color='#AAAAAA')
    ax.grid(which='major', color='black')


if __name__ == '__main__':
    main()
