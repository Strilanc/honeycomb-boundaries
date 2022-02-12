import pathlib
import sys
from typing import List, Tuple, Dict, Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from hcb.artifacts.make_lambda_plots import DesiredLineFit, project_intersection_of_both_observables
from hcb.artifacts.make_threshold_plots import make_threshold_plots
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
        ('planar honeycomb code\n(correlated MWPM decoding)', 'bounded_honeycomb_memory', 'internal_correlated', 'H'),
        ('planar honeycomb code\n(correlated MWPM decoding)', 'bounded_honeycomb_memory', 'internal_correlated', 'V'),
        ('planar honeycomb code\n(MWPM decoding)', 'bounded_honeycomb_memory', 'internal', 'H'),
        ('planar honeycomb code\n(MWPM decoding)', 'bounded_honeycomb_memory', 'internal', 'V'),
    }
    groups = {
        gate_set_caption: [
            DesiredLineFit(
                legend_caption=layout_caption,
                marker=MARKERS[k],
                color=COLORS[k],
                observable=obs,
                filter_circuit_style=f"{gate_set_prefix}_{gate_set_suffix}",
                filter_decoder=decoder,
            )
            for k, (layout_caption, gate_set_prefix, decoder, obs) in enumerate(layouts)
        ]
        for gate_set_caption, gate_set_suffix in gate_sets.items()
    }

    fig, _ = make_threshold_plots(data=all_data, groups=groups)
    fig.set_size_inches(18, 18)
    fig.savefig(OUT_DIR / "threshold_breakdown.png", bbox_inches='tight', dpi=200)

    plt.show()


if __name__ == '__main__':
    main()
