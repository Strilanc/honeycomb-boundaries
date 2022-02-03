import collections
from typing import DefaultDict

from hcb.codes.honeycomb.layout import HoneycombLayout

def produce_code_distance_latex_table() -> str:
    entries = collections.defaultdict(list)
    widths = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    heights = [3 * k + 3 for k in range(len(widths))]
    assert len(widths) == len(heights)
    y = 0
    for case in range(2):
        vals = heights if case == 0 else widths
        if case == 0:
            entries[y*1j].append('using w>>h')
        else:
            entries[y * 1j].append('using h>>w')
        for x, v in enumerate(vals):
            if case == 0:
                entries[x + 1 + y*1j].append(f'$h={v}$')
            else:
                entries[x + 1 + y*1j].append(f'$w={v}$')
        y += 1
        for sheared in [False, True]:
            for gate_set in ['SD6', 'SI1000', 'EM3_v2']:
                s = '(sheared) ' if sheared else ''
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
                    if sheared and data_width % 2 == 1:
                        d = "N/A"
                    else:
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
                        err = circuit.shortest_graphlike_error(ignore_ungraphlike_errors=True)
                        d = str(len(err))
                        print(layout.to_decoding_desc(decoder="internal"))
                    d = d.rjust(4)
                    entries[x + 1 + 1j*y].append(d)
        y += 1
    max_x = int(max(e.real for e in entries))
    max_y = int(max(e.imag for e in entries))
    entry_counts: DefaultDict[int, int] = collections.defaultdict(int)
    for k, v in entries.items():
        entry_counts[int(k.imag)] = max(entry_counts[int(k.imag)], len(v))
    out = '\\begin{tabular}{| r |'
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

    return out


if __name__ == '__main__':
    print(produce_code_distance_latex_table())
