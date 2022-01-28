import pathlib
import sys

from hcb.tools.analysis.collecting import read_recorded_data, ProblemShotData
from hcb.tools.analysis.plotting import plot_data

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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

    fig = plot_thresholds(all_data, focused=True)
    fig.set_size_inches(13, 10)
    fig.savefig(OUT_DIR / "threshold.pdf", bbox_inches='tight')
    fig.savefig(OUT_DIR / "threshold.png", bbox_inches='tight', dpi=200)

    fig2 = plot_thresholds(all_data, focused=False)
    fig2.set_size_inches(13, 20)
    fig2.savefig(OUT_DIR / "threshold_all.pdf", bbox_inches='tight')
    fig2.savefig(OUT_DIR / "threshold_all.png", bbox_inches='tight', dpi=200)

    plt.show()


def plot_thresholds(all_data: ProblemShotData, focused: bool) -> plt.Figure:
    styles = {
        "SD6": [
            # ("surface_SD6", "X", "internal"),
            # ("surface_SD6", "Z", "internal"),
            # ("surface_SD6", "X", "internal_correlated"),
            # ("surface_SD6", "Z", "internal_correlated"),
            ("honeycomb_SD6", "H", "internal"),
            ("honeycomb_SD6", "V", "internal"),
            ("honeycomb_SD6", "H", "internal_correlated"),
            ("honeycomb_SD6", "V", "internal_correlated"),
            ("bounded_honeycomb_memory_SD6", "H", "internal"),
            ("bounded_honeycomb_memory_SD6", "V", "internal"),
            ("bounded_honeycomb_memory_SD6", "H", "internal_correlated"),
            ("bounded_honeycomb_memory_SD6", "V", "internal_correlated"),
        ],
        "SI1000": [
            # ("surface_SI1000", "X", "internal"),
            # ("surface_SI1000", "Z", "internal"),
            # ("surface_SI1000", "X", "internal_correlated"),
            # ("surface_SI1000", "Z", "internal_correlated"),
            ("honeycomb_SI1000", "H", "internal"),
            ("honeycomb_SI1000", "V", "internal"),
            ("honeycomb_SI1000", "H", "internal_correlated"),
            ("honeycomb_SI1000", "V", "internal_correlated"),
            ("bounded_honeycomb_memory_SI1000", "H", "internal"),
            ("bounded_honeycomb_memory_SI1000", "V", "internal"),
            ("bounded_honeycomb_memory_SI1000", "H", "internal_correlated"),
            ("bounded_honeycomb_memory_SI1000", "V", "internal_correlated"),
        ],
        "EM3": [
            # None,
            # None,
            # None,
            # None,
            ("honeycomb_EM3_v2", "H", "internal"),
            ("honeycomb_EM3_v2", "V", "internal"),
            ("honeycomb_EM3_v2", "H", "internal_correlated"),
            ("honeycomb_EM3_v2", "V", "internal_correlated"),
            ("bounded_honeycomb_memory_EM3_v2", "H", "internal"),
            ("bounded_honeycomb_memory_EM3_v2", "V", "internal"),
            ("bounded_honeycomb_memory_EM3_v2", "H", "internal_correlated"),
            ("bounded_honeycomb_memory_EM3_v2", "V", "internal_correlated"),
        ],
    }
    if focused:
        styles = {
            "SD6": [
                # ("surface_SD6", "Z", "internal_correlated"),
                ("honeycomb_SD6", "H", "internal_correlated"),
                ("bounded_honeycomb_memory_SD6", "H", "internal_correlated"),
            ],
            "SI1000": [
                # ("surface_SI1000", "Z", "internal_correlated"),
                ("honeycomb_SI1000", "H", "internal_correlated"),
                ("bounded_honeycomb_memory_SI1000", "H", "internal_correlated"),
            ],
            "EM3": [
                # None,
                ("honeycomb_EM3_v2", "H", "internal_correlated"),
                ("bounded_honeycomb_memory_EM3_v2", "H", "internal_correlated"),
            ],
        }

    all_groups = all_data.grouped_by(lambda e: (e.circuit_style, e.preserved_observable, e.decoder))

    fig = plt.figure()
    ncols = len(styles)
    nrows = len(styles["SD6"])
    gs = fig.add_gridspec(ncols=ncols, nrows=nrows, hspace=0.075, wspace=0.05)
    axs = gs.subplots(sharex=True, sharey=True)
    used = set()
    for col, (name, cases) in enumerate(styles.items()):
        for row, style_obs_decoder in enumerate(cases):
            ax: plt.Axes = axs[row][col]
            if style_obs_decoder is None:
                ax.axis('off')
                continue
            used.add((row, col))
            style_data = all_groups.get(style_obs_decoder, ProblemShotData({}))
            ax: plt.Axes = axs[row][col]
            plot_data(
                style_data,
                title=name,
                ax=ax,
                fig=fig,
                legend=False,
                x_max=0.03,
                marker_offset=4 if style_obs_decoder[1] in "XZ" else 0)

    a1, b1 = axs[0][0].get_legend_handles_labels()
    a2, b2 = axs[-1][0].get_legend_handles_labels()
    axs[0][-1].legend(
        [
            mpatches.Patch(color='white', label='Surface Code Sizes:'),
            *a1,
            mpatches.Patch(color='white', label='Honeycomb Code Sizes:'),
            *a2,
        ],
        [
            "Surface Code Sizes:",
            *b1,
            "Honeycomb Code Sizes:",
            *b2,
        ],
        loc="upper left",
    )
    for k in range(nrows):
        style, obs, decoder = styles["SD6"][k]
        if obs == "H":
            obs = "Horizontal"
        if obs == "V":
            obs = "Vertical"
        style = style.split("_")[0]
        style = style.capitalize()
        if "correlated" in decoder:
            style += " (Correlated)"
        else:
            style += " (Standard)"
        axs[k][0].set_ylabel(f"{style}\n{obs} Observable\nCode Cell Error Rate")
    for row in range(nrows):
        for col in range(ncols):
            if (row + 1, col) in used:
                axs[row][col].set_xlabel("")
            if (row - 1, col) in used:
                axs[row][col].set_title("")
            if (row, col - 1) in used:
                axs[row][col].set_ylabel("")
    for ax_row in axs:
        for ax in ax_row:
            ax.yaxis.set_ticks_position('both')
    return fig


if __name__ == '__main__':
    main()
