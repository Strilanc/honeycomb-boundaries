import pathlib
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from hcb.tools.analysis.collecting import MultiStats

OUT_DIR = pathlib.Path(__file__).parent.parent.parent.parent / "out"


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

    all_data = MultiStats.from_recorded_data(*csvs)

    fig = plot_detection_fraction(all_data)
    fig.set_size_inches(15, 5)
    fig.savefig(OUT_DIR / "grouped_detection_fractions.png", bbox_inches='tight', dpi=200)

    plt.show()


def plot_detection_fraction(all_data: MultiStats) -> plt.Figure:
    included_styles = {
        'bounded_honeycomb_memory_SD6': ("SD6", "Planar Honeycomb Code"),
        'bounded_honeycomb_memory_SI1000': ("SI1000", "Planar Honeycomb Code"),
        "bounded_honeycomb_memory_EM3_v2": ("EM3", "Planar Honeycomb Code"),
    }
    known_styles = {
        "surface_SD6",
        "surface_SI1000",
        "honeycomb_SD6",
        "bounded_honeycomb_memory_EM3_v1",
        "honeycomb_SI1000",
        "honeycomb_EM3_v2",
        'bounded_honeycomb_memory_EM3_v2',
        'bounded_honeycomb_memory_SI1000',
        'bounded_honeycomb_memory_sheared_EM3_v1',
        'bounded_honeycomb_memory_sheared_EM3_v2',
        'bounded_honeycomb_memory_sheared_SD6',
        'bounded_honeycomb_memory_sheared_SI1000',
        'honeycomb_EM3',
        'bounded_honeycomb_memory_SD6',
    }

    p2i = {p: i for i, p in enumerate(sorted(set(e.noise for e in all_data.data.keys())))}
    all_groups = all_data.grouped_by(lambda e: e.circuit_style)
    missed = sorted(key for key in all_groups.keys()if key not in known_styles)
    if missed:
        raise NotImplementedError(repr(missed))

    names = [
        "Xa",
        "Ya",
        "Za",
        "Xb",
        "Zb",
        "Yb",
    ]
    fig = plt.figure()
    ncols = len(included_styles) // 1
    nrows = 1
    gs = fig.add_gridspec(ncols=ncols + 1, nrows=nrows, hspace=0.05, wspace=0.1)
    axs = gs.subplots(sharex=True, sharey=True)
    markers = "ov*sp^<>8P+xXDd|"
    colors = list(mcolors.TABLEAU_COLORS) * 3
    for k2, style in enumerate(included_styles):
        col = k2
        ax: plt.Axes = axs[col]
        style_data = all_groups.get(style, MultiStats({})).grouped_by(lambda e: e.noise)
        for noise, case_data in style_data.items():
            xs = []
            ys = []
            for k, det_data in case_data.grouped_by(lambda e: e.decoder).items():
                index = int(k.split("_")[-1])
                assert len(det_data.data) == 1
                xs.append(index)
                ys.append(det_data.merged_total().logical_error_rate)
            order = p2i[noise]
            xs += [e + 6 for e in xs]
            ys *= 2
            ax.plot(xs, ys, label=str(noise), marker=markers[order], color=colors[order])
        ax.set_title(included_styles[style][0])
        if col == 0:
            ax.set_ylabel(included_styles[style][1] + "\nDetection Fraction")
        ax.set_ylim(0, 0.5)
        ax.set_xlim(0, 5)
        ax.set_xticks(range(len(names) * 2))
        ax.set_xticklabels(names * 2)
    for x in range(ncols):
        axs[x].grid()
        axs[x].set_yticks([])

    axs[0].set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    axs[0].set_yticklabels(["0%", "10%", "20%", "30%", "40%", "50%"])
    axs[-1].axis('off')
    a, b = axs[-2].get_legend_handles_labels()
    axs[-1].legend(a[::-1], b[::-1], loc='upper left', title="Physical Error Rate")
    for x in range(ncols):
        axs[x].set_xlabel("Edge Measurement Step")

    return fig


if __name__ == '__main__':
    main()
