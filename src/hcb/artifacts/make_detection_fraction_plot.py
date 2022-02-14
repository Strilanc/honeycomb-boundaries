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
    fig.savefig(OUT_DIR / "detection_fractions.png", bbox_inches='tight', dpi=200)

    plt.show()


def plot_detection_fraction(all_data: MultiStats) -> plt.Figure:
    included_styles = {
        'honeycomb_SD6': ("SD6", "Periodic Honeycomb Code"),
        'honeycomb_SI1000': ("SI1000", "Periodic Honeycomb Code"),
        "honeycomb_EM3_v2": ("EM3", "Periodic Honeycomb Code"),
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

    fig = plt.figure()
    ncols = len(included_styles) // 2
    nrows = 2
    gs = fig.add_gridspec(ncols=ncols + 1, nrows=nrows, hspace=0.05, wspace=0.1)
    axs = gs.subplots(sharex=True, sharey=True)
    markers = "ov*sp^<>8P+xXDd|"
    colors = list(mcolors.TABLEAU_COLORS) * 3
    for k2, style in enumerate(included_styles):
        col = k2 % ncols
        row = k2 // ncols
        ax: plt.Axes = axs[row, col]
        style_data = all_groups.get(style, MultiStats({})).grouped_by(lambda e: e.noise)
        for noise, case_data in style_data.items():
            xs = []
            ys = []
            for k, v in case_data.data.items():
                xs.append(k.data_width)
                ys.append(v.logical_error_rate)
            order = p2i[noise]
            ax.plot(xs, ys, label=str(noise), marker=markers[order], color=colors[order])
        if row == 0:
            ax.set_title(included_styles[style][0])
        if col == 0:
            ax.set_ylabel(included_styles[style][1] + "\nDetection Fraction")
        ax.set_ylim(0, 0.5)
        ax.set_xlim(0, 20)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_xticklabels(["0", "5", "10", "15", "20"], rotation=90)
    for x in range(ncols):
        for y in range(nrows):
            axs[y, x].grid()
            axs[y, x].set_yticks([])

    for y in range(nrows):
        axs[y, 0].set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
        axs[y, 0].set_yticklabels(["0%", "10%", "20%", "30%", "40%", "50%"])
        axs[y, -1].axis('off')
    a, b = axs[1, -2].get_legend_handles_labels()
    axs[0, -1].legend(a[::-1], b[::-1], loc='upper left', title="Physical Error Rate")
    axs[1, -1].set_visible(False)
    for x in range(ncols):
        axs[-1, x].set_xlabel("Patch Width")

    return fig


if __name__ == '__main__':
    main()
