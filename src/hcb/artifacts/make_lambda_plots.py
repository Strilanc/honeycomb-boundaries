import dataclasses
import math
import pathlib
import sys
from typing import List, Tuple, Dict

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from hcb.tools.analysis.collecting import MultiStats
from hcb.tools.analysis.probability_util import least_squares_slope_range

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

    fig2, _ = make_lambda_plots(all_data, groups)
    fig2.set_size_inches(18, 6)
    fig2.savefig(OUT_DIR / "lambda.png", bbox_inches='tight', dpi=200)

    plt.show()


@dataclasses.dataclass
class DesiredLineFit:
    legend_caption: str
    marker: str
    color: str
    filter_circuit_style: str
    filter_decoder: str
    observable: str


def fill_in_single_lambda_plot(
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

            low, best, high = slope_range_of_line_fit(linefit_xs, linefit_ys)
            xs.append(noise)
            ys_low.append(low)
            ys.append(best)
            ys_high.append(high)
        ax.plot(xs, ys, marker=group.marker, label=group.legend_caption, color=group.color, zorder=10)
        ax.fill_between(xs, ys_low, ys_high, alpha=0.2, color=group.color)

        ax.set_xlabel("Physical Error Rate")
        ax.set_xlim(1e-4, 2e-2)
        ax.set_ylim(1, 100)
        ax.loglog()
        ax.grid(which='minor', color='#AAAAAA')
        ax.grid(which='major', color='black')


def make_lambda_plots(
        data: MultiStats,
        groups: Dict[str, List[DesiredLineFit]]):
    data = project_intersection_of_both_observables(data)
    grouped = data.grouped_by(lambda e: (e.circuit_style, e.decoder))

    fig, axs = plt.subplots(1, len(groups) + 1)
    for k, (name, gs) in enumerate(groups.items()):
        fill_in_single_lambda_plot(grouped, gs, axs[k])
        axs[k].set_title(name)
    axs[0].set_ylabel("Suppression per Code Distance Step (λ)")
    for k in range(1, len(groups)):
        axs[k].set_yticklabels([])

    axs[-1].legend(*axs[-2].get_legend_handles_labels(), loc="upper left")
    axs[-1].get_xaxis().set_ticks([])
    axs[-1].get_yaxis().set_ticks([])
    axs[-1].spines['top'].set_visible(False)
    axs[-1].spines['right'].set_visible(False)
    axs[-1].spines['bottom'].set_visible(False)
    axs[-1].spines['left'].set_visible(False)

    return fig, axs


def project_intersection_of_both_observables(stats: MultiStats) -> MultiStats:
    obs_groups = stats.grouped_by(lambda e: e.preserved_observable)
    known_keys = {k.with_changes(preserved_observable='combo') for k, v in stats.data.items()}
    result = MultiStats()
    h_group = obs_groups.get('H', MultiStats())
    v_group = obs_groups.get('V', MultiStats())
    x_group = obs_groups.get('X', MultiStats())
    z_group = obs_groups.get('Z', MultiStats())
    epr_group = obs_groups.get('EPR', MultiStats())
    for k in known_keys:
        h = h_group.data.get(k.with_changes(preserved_observable='H'))
        v = v_group.data.get(k.with_changes(preserved_observable='V'))
        x = x_group.data.get(k.with_changes(preserved_observable='X'))
        z = z_group.data.get(k.with_changes(preserved_observable='Z'))
        epr = epr_group.data.get(k.with_changes(preserved_observable='EPR'))
        if h is None and v is None:
            h, v = x, z

        if h is not None and v is not None:
            result.data[k] = h.extrapolate_intersection(v)
        elif h is not None:
            print(f"WARNING EXTRAPOLATING V OBSERVABLE DATA POINT FOR {k}", file=sys.stderr, flush=True)
            result.data[k] = h
        elif v is not None:
            print(f"WARNING EXTRAPOLATING H OBSERVABLE DATA POINT FOR {k}", file=sys.stderr, flush=True)
            result.data[k] = v
        elif epr is not None:
            result.data[k] = epr
        else:
            raise NotImplementedError(f"Missing observable group: from {sorted(obs_groups.keys())} get {k}")
    return result


def slope_range_of_line_fit(xs: List[float], ys: List[float]) -> Tuple[float, float, float]:
    def slope_to_lambda(s: float) -> float:
        return 1 / math.exp(s) ** 2

    if len(xs) <= 1:
        raise ValueError("len(xs) <= 1")
    slopes = least_squares_slope_range(xs=xs, ys=ys, cost_increase=1)
    a, b, c = tuple(slope_to_lambda(s) for s in slopes)
    return a, b, c


if __name__ == '__main__':
    main()
