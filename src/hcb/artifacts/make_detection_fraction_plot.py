import pathlib
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from hcb.tools.analysis.collecting import read_recorded_data, ProblemShotData

OUT_DIR = pathlib.Path("../../../out/").resolve()


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

    all_data = read_recorded_data(*csvs)

    fig = plot_detection_fraction(all_data)
    fig.set_size_inches(15, 5)
    fig.savefig(OUT_DIR / "detection_fractions.pdf", bbox_inches='tight')
    fig.savefig(OUT_DIR / "detection_fractions.png", bbox_inches='tight', dpi=200)


    plt.show()


def plot_detection_fraction(all_data: ProblemShotData) -> plt.Figure:
    included_styles = {
        'honeycomb_SI1000': 'SI1000\nToric',
        'bounded_honeycomb_memory_SI1000': 'SI1000\nBounded',
        'honeycomb_SD6': 'SD6\nToric',
        'bounded_honeycomb_memory_SD6': 'SD6\nBounded',
        "honeycomb_EM3_v2": "EM3\nToric",
        "bounded_honeycomb_memory_EM3_v2": "EM3\nBounded",
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
    ncols = len(included_styles)
    nrows = 1
    gs = fig.add_gridspec(ncols=ncols + 1, nrows=nrows, hspace=0.05, wspace=0.1)
    axs = gs.subplots(sharex=True, sharey=True)
    markers = "ov*sp^<>8P+xXDd|"
    colors = list(mcolors.TABLEAU_COLORS) * 3
    for col, style in enumerate(included_styles):
        ax: plt.Axes = axs[col]
        style_data = all_groups.get(style, ProblemShotData({})).grouped_by(lambda e: e.noise)
        for noise, case_data in style_data.items():
            xs = []
            ys = []
            for k, v in case_data.data.items():
                xs.append(k.code_distance)
                ys.append(v.logical_error_rate)
            order = p2i[noise]
            ax.plot(xs, ys, label=str(noise), marker=markers[order], color=colors[order])
        ax.set_title(included_styles[style])
        ax.set_ylim(0, 0.5)
        ax.set_xlim(0, 20)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_xticklabels(["0", "5", "10", "15", "20"], rotation=90)
    axs[0].set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    axs[0].set_yticklabels(["0%", "10%", "20%", "30%", "40%", "50%"])
    axs[-1].axis('off')
    a, b = axs[-2].get_legend_handles_labels()
    axs[-1].legend(a[::-1], b[::-1], loc='upper left', title="Physical Error Rate")
    axs[0].set_ylabel("Detection Fraction")
    for ax in axs:
        ax.set_xlabel("Code Distance")
        ax.grid()

    return fig


if __name__ == '__main__':
    main()
