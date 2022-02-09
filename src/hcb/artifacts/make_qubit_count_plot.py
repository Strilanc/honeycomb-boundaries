import pathlib

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from hcb.codes.honeycomb.layout import HoneycombLayout

OUT_DIR = pathlib.Path(__file__).parent.parent.parent.parent / "out"

MARKERS = "ov*sp^<>8PhH+xXDd|" * 100
COLORS = list(mcolors.TABLEAU_COLORS) * 3


def main():
    gate_sets = {
        'SD6 or SI1000': 'SD6',
        'EM3': 'EM3_v2',
    }
    ax: plt.Axes
    fig: plt.Figure
    fig, ax = plt.subplots(1, 1)
    markers = "ov*sp^<>8PhH+xXDd|" * 100
    for k, (name, key) in enumerate(gate_sets.items()):
        xs = []
        ys = []
        for d in range(1, 31):
            w, h = HoneycombLayout.unsheared_size_for_code_distance(distance=d, gate_set=key)
            layout = HoneycombLayout(
                data_width=w,
                data_height=h,
                rounds=10,
                noise_level=0.001,
                noisy_gate_set=key,
                tested_observable='EPR',
                sheared=False,
            )
            xs.append(d)
            ys.append(layout.num_used_qubits())
        ax.plot(xs, ys, label=name, marker=markers[k])
    ax.legend()
    ax.grid()
    ax.set_xlabel("Graphlike Code Distance")
    ax.set_ylabel("Total Physical Qubits")
    ax.set_ylim(0, 5000)
    ax.set_xlim(0, 30)

    fig.set_size_inches(13, 7)
    fig.savefig(OUT_DIR / "qubit_counts.png", bbox_inches='tight', dpi=200)

    plt.show()


if __name__ == '__main__':
    main()
