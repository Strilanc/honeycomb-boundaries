import collections
from typing import DefaultDict

from hcb.codes.honeycomb.layout import HoneycombLayout


def produce_code_distance_latex_table() -> str:
    entries = collections.defaultdict(list)
    widths = [2, 4, 6, 8, 10, 12, 14]
    heights = [3, 6, 9, 12, 15, 18, 21]
    y = 0
    for case in range(2):
        vals = heights if case == 0 else widths
        if case == 0:
            entries[y*1j].append('H-vulnerable layout (w=2h)')
        else:
            entries[y * 1j].append('V-vulnerable layout (h=3w)')
        for x, v in enumerate(vals):
            if case == 0:
                entries[x + 1 + y*1j].append(f'$h={v}$')
            else:
                entries[x + 1 + y*1j].append(f'$w={v}$')
        y += 1
        for gate_set in ['EM3_v2', 'SD6', 'SI100']:
            for sheared in [True, False]:
                s = '(tilt) ' if sheared else ''
                g = gate_set.replace('EM3_v2', 'EM3')
                entries[y * 1j].append(f'{s}{g}')
                for x, v in enumerate(vals):
                    if case == 0:
                        data_width = v * 2
                        data_height = v
                        obs = 'H'
                    else:
                        data_width = v
                        data_height = v * 3 - 3
                        obs = 'V'
                    layout = HoneycombLayout(
                        data_width=data_width,
                        data_height=data_height,
                        rounds=4,
                        noise_level=0.001,
                        noisy_gate_set=gate_set,
                        tested_observable=obs,
                        sheared=sheared,
                    )
                    circuit = layout.noisy_circuit()
                    dem = circuit.detector_error_model(decompose_errors=False)
                    err = dem.shortest_graphlike_error(ignore_ungraphlike_errors=True)
                    d = str(len(err))
                    d = d.rjust(4)
                    entries[x + 1 + 1j*y].append(d)
        y += 1
    max_x = int(max(e.real for e in entries))
    max_y = int(max(e.imag for e in entries))
    entry_counts: DefaultDict[int, int] = collections.defaultdict(int)
    for k, v in entries.items():
        entry_counts[int(k.imag)] = max(entry_counts[int(k.imag)], len(v))
    out = '\\begin{table}\n'
    out += '\\centering\n'
    out += '\\begin{tabular}{| r |'
    for x in range(1, max_x + 1):
        out += ' c |'
    out += '}\n'
    for y in range(max_y + 1):
        if entry_counts[y] == 1:
            out += '    \\hline\n'
        for y2 in range(entry_counts[y]):
            out += '    '
            for x in range(max_x + 1):
                r = entries[x + 1j*y]
                if x:
                    out += ' & '
                out += (r[y2] if y2 < len(r) else '').rjust(16)
            out += ' \\\\\n'
        if entry_counts[y] == 1:
            out += '    \\hline\n'
    out += '    \\hline\n'
    out += '\\end{tabular}\n'
    out += '\\label{tbl:graphlike_distances}'
    out += r'''
\caption{
    Horizontal and vertical graphlike code distances of bounded honeycomb patches under circuit noise.
    Computed using \emph{stim.Circuit.shortest\_graphlike\_error}.
    A ``graphlike error" is an error that produces at most two detection events.
    The graphlike code distance is the size of the smallest set of graphlike errors
    that produce an undetectable logical error.
    The true code distance may be smaller, but many issues can be caught by computing
    the graphlike code distance.
    For example, this is how we realized that a 2:3 aspect ratio was not optimal for
    the SD6 layout, and that tilting could reduce the code distance.
}
'''
    out += '\\end{table}'

    return out


if __name__ == '__main__':
    print(produce_code_distance_latex_table())
