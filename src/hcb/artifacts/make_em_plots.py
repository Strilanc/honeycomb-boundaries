import math
import pathlib
import sys
from typing import List, Tuple, Dict

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from scipy.stats import linregress

from hcb.artifacts.make_lambda_plots import DesiredLineFit, project_intersection_of_both_observables, fill_in_single_lambda_plot
from hcb.artifacts.make_teraquop_plots import fill_in_single_teraquop_plot
from hcb.tools.analysis.collecting import MultiStats, DecodingProblemDesc

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
        'EM3': 'EM3_v2',
        'SDEM3': 'SDEM3',
        'SIEM3000': 'SIEM3000',
    }
    gate_set_prefix = 'bounded_honeycomb_memory'
    decoder = 'internal_correlated'
    group = [
            DesiredLineFit(
                legend_caption=gate_set_caption,
                marker=MARKERS[k],
                color=COLORS[k],
                filter_circuit_style=f"{gate_set_prefix}_{gate_set_suffix}",
                filter_decoder=decoder,
                observable="combo",
            )
        for k, (gate_set_caption, gate_set_suffix) in enumerate(gate_sets.items())
    ]

    fig, _ = make_em_plots(all_data, group)
    fig.set_size_inches(18, 6)
    fig.savefig(OUT_DIR / "em.png", bbox_inches='tight', dpi=200)

    plt.show()


def make_em_plots(
        data: MultiStats,
        group: List[DesiredLineFit]
):

    data = project_intersection_of_both_observables(data)
    grouped = data.grouped_by(lambda e: (e.circuit_style, e.decoder))

    axs: List[plt.Axes]
    fig, axs = plt.subplots(1, 3)  # Lambda, Terraquops, empty for legend

    fill_in_single_lambda_plot(grouped, group, axs[0])
    axs[0].set_ylabel("Suppression per Code Distance Step (λ)")
    axs[0].yaxis.label.set_fontsize(14)

    fill_in_single_teraquop_plot(grouped, group, axs[1])
    axs[1].set_ylabel("Teraquop Qubit Count")
    axs[1].yaxis.label.set_fontsize(14)

    axs[-1].legend(*axs[-2].get_legend_handles_labels(), loc="upper left")
    axs[-1].get_xaxis().set_ticks([])
    axs[-1].get_yaxis().set_ticks([])
    axs[-1].spines['top'].set_visible(False)
    axs[-1].spines['right'].set_visible(False)
    axs[-1].spines['bottom'].set_visible(False)
    axs[-1].spines['left'].set_visible(False)

    return fig, axs


if __name__ == '__main__':
    main()
