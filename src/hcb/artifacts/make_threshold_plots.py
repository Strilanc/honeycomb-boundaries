import pathlib
import sys
from typing import List, Tuple, Dict, Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from hcb.artifacts.make_lambda_plots import DesiredLineFit, project_intersection_of_both_observables
from hcb.tools.analysis.collecting import MultiStats

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
        'periodic honeycomb code\n(correlated MWPM decoding)': ('honeycomb', 'internal_correlated'),
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
            )
            for k, (layout_caption, (gate_set_prefix, decoder)) in enumerate(layouts.items())
        ]
        for gate_set_caption, gate_set_suffix in gate_sets.items()
    }

    fig, _ = make_threshold_plots(all_data, groups)
    fig.set_size_inches(24, 8)
    fig.savefig(OUT_DIR / "threshold.png", bbox_inches='tight', dpi=200)

    plt.show()


def make_threshold_plots(
        data: MultiStats,
        groups: Dict[str, List[DesiredLineFit]]):
    data = project_intersection_of_both_observables(data)
    grouped = data.grouped_by(lambda e: (e.circuit_style, e.decoder))
    sizes = sorted({(desc.data_width, desc.data_height, desc.rounds)
                    for desc in data.data.keys()}, reverse=True)
    size_styles = {
        noise: (COLORS[k], MARKERS[k])
        for k, noise in enumerate(sizes)
    }

    fig, axs = plt.subplots(2, len(groups) + 1)
    for k, (name, gs) in enumerate(groups.items()):
        for y, g in enumerate(gs):
            fill_in_threshold_plot(grouped, g, axs[y, k], size_styles)
            if k == 0:
                axs[y, 0].set_ylabel(f"{g.legend_caption}\nLogical Error Rate")
                axs[y, 0].yaxis.label.set_fontsize(14)
        axs[0, k].set_title(name)
        axs[-1, k].set_xlabel("Physical Error Rate")
        axs[-1, k].xaxis.label.set_fontsize(14)

    for y in [0, 1]:
        labels = []
        handles = []
        for x in range(3):
            a, b = axs[y, x].get_legend_handles_labels()
            for k in range(len(a)):
                if b[k] not in handles:
                    labels.append(a[k])
                    handles.append(b[k])
        axs[y, -1].legend(labels, handles, loc="upper left")
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
    size_styles: Dict[Tuple[float, float, float], Tuple[Any, Any]],
):
    stats = grouped_projected_data.get((group.filter_circuit_style, group.filter_decoder), MultiStats({}))
    for size_key, noise_stats in stats.grouped_by(lambda e: (e.data_width, e.data_height, e.rounds)).items():
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
        color, marker = size_styles[size_key]
        ax.plot(xs, ys, marker=marker, label=f'{size_key[0]}x{size_key[1]} ({size_key[2]} rounds)', color=color, zorder=10)
        ax.fill_between(xs, ys_low, ys_high, alpha=0.3, color=color)

    ax.set_xlim(1e-4, 1e-1)
    ax.set_ylim(1e-6, 1e-0)
    ax.loglog()
    ax.grid(which='minor', color='#AAAAAA')
    ax.grid(which='major', color='black')


if __name__ == '__main__':
    main()
