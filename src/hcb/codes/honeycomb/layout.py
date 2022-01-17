import collections
import dataclasses
import functools
import pathlib
from typing import List, Tuple, Iterable, FrozenSet, Optional, Dict, AbstractSet

import stim

from hcb.tools.analysis.collecting import DecodingProblem, DecodingProblemDesc
from hcb.tools.gen.circuit_canvas import complex_key
from hcb.tools.gen.measurement_tracker import MeasurementTracker, Prev
from hcb.tools.gen.stabilizer_plan import StabilizerPlan, StabilizerPlanElement
from hcb.tools.gen.viewer import stim_circuit_html_viewer


HEX_DATA_OFFSETS = (
    0,
    1,
    1 + 1j,
    1 + 2j,
    1j,
    2j,
)
HEX_MEASURE_OFFSETS = (
    0.5,
    0.5j,
    1.5j,
    1 + 0.5j,
    1 + 1.5j,
    0.5 + 2j,
)
EDGE_BASIS_SEQUENCE = 'YXZ'
EDGE_MEASUREMENT_SEQUENCE = 'XYZXZY'


def find_hex_comparison(previous_measurements: str) -> Optional[Tuple[str, Tuple[Prev, ...]]]:
    if len(previous_measurements) < 2:
        return None
    def piece(pos: int) -> Prev:
        char = previous_measurements[pos]
        return Prev(char, offset=sum(c == char for c in previous_measurements[pos + 1:]))
    match = set(previous_measurements[-2:])
    remainder, = set('XYZ') - match
    n = len(previous_measurements)
    for k in range(n - 2)[::-1]:
        if set(previous_measurements[k:k+2]) == match:
            return remainder, (
                piece(n-1),
                piece(n-2),
                piece(k),
                piece(k+1),
            )
    return None


@dataclasses.dataclass(unsafe_hash=True, frozen=True)
class HoneycombHex:
    top_left: complex
    data_qubits: Tuple[complex, ...]
    measurement_qubits: Tuple[complex, ...]

    def __post_init__(self):
        assert (self.top_left.real + self.top_left.imag) % 2 == 0

    @property
    def basis(self) -> str:
        return EDGE_BASIS_SEQUENCE[(2 + int(self.top_left.imag)) % 3]

    @property
    def center(self) -> complex:
        return self.top_left + 0.5 + 1j

    @staticmethod
    def from_top_left(top_left: complex, present_qubits: Optional[AbstractSet[complex]] = None) -> 'HoneycombHex':
        measurement_qubits = tuple(top_left + d for d in HEX_MEASURE_OFFSETS)
        data_qubits = tuple(top_left + d for d in HEX_DATA_OFFSETS)
        if present_qubits is not None:
            data_qubits = tuple(q for q in data_qubits if q in present_qubits)
            measurement_qubits = tuple(q for q in measurement_qubits if q in present_qubits)
        return HoneycombHex(top_left=top_left,
                            data_qubits=data_qubits,
                            measurement_qubits=measurement_qubits)

    def as_plan_element(self) -> StabilizerPlanElement:
        return StabilizerPlanElement(
            bases=(self.basis,) * len(self.data_qubits),
            data_qubit_order=self.data_qubits,
            measurement_qubit=sum(self.measurement_qubits) / len(self.measurement_qubits),
            measurement_qubit_is_artificial=True,
        )


class HoneycombLayout:
    def __init__(self, *, data_width: int, data_height: int, rounds: int, noise: float):
        self.data_width = data_width
        self.data_height = data_height
        self.rounds = rounds
        self.noise = noise
        assert self.data_width % 2 == 0
        assert self.data_height % 3 == 0
        assert self.data_width * 3 >= self.data_height

    @staticmethod
    def from_code_distance(distance: int, rounds: int, noise: float):
        assert distance % 2 == 0
        return HoneycombLayout(
            data_width=distance + 2,
            data_height=(distance // 2 + 1) * 3,
            rounds=rounds,
            noise=noise)

    @functools.cached_property
    def measurement_qubit_set(self) -> FrozenSet[complex]:
        return frozenset(e.measurement_qubit for e in self.edge_plan.elements)

    @functools.cached_property
    def leaf_element_set(self) -> FrozenSet[StabilizerPlanElement]:
        return frozenset(
            e
            for e in self.edge_plan.elements
            if sum(q is not None for q in e.data_qubit_order) == 1
        )

    @functools.cached_property
    def data_to_elements_dict(self) -> Dict[complex, Tuple[StabilizerPlanElement, ...]]:
        result: Dict[complex, List[StabilizerPlanElement]] = {}
        for e in self.edge_plan.elements:
            for q in e.data_coords_set():
                result.setdefault(q, []).append(e)
        return {k: tuple(v) for k, v in result.items()}

    @functools.cached_property
    def measure_to_element_dict(self) -> Dict[complex, StabilizerPlanElement]:
        result: Dict[complex, StabilizerPlanElement] = {}
        for e in self.edge_plan.elements:
            result[e.measurement_qubit] = e
        return result

    @functools.cached_property
    def leaf_data_to_leaf_element_dict(self) -> Dict[complex, StabilizerPlanElement]:
        return {
            q: e
            for e in self.leaf_element_set
            for q in e.data_coords_set()
        }

    @functools.cached_property
    def edge_plan(self) -> StabilizerPlan:
        elements = []
        for x in range(-2, self.data_width * 2 + 2):
            for y in range(-2, self.data_height + 2):
                # Vertical edge.
                c = x + 1j * y
                h_basis = EDGE_BASIS_SEQUENCE[int(y) % 3]
                elements.append(StabilizerPlanElement(
                    bases=(h_basis,) * 2,
                    measurement_qubit=c + 0.5j,
                    data_qubit_order=(
                        c,
                        c + 1j,
                    ),
                ))

                if (x + y) & 1 == 0:
                    v_basis = EDGE_BASIS_SEQUENCE[int(y) % 3 - 2]
                    elements.append(StabilizerPlanElement(
                        bases=(v_basis,) * 2,
                        measurement_qubit=c + 0.5,
                        data_qubit_order=(
                            c,
                            c + 1,
                        ),
                    ))

        # Make boundary cuts.
        kept_elements = []
        for e in elements:
            # Cut along top boundary.
            if e.measurement_qubit.imag < -0.5:
                continue
            # Cuts along bottom boundary.
            if e.measurement_qubit.imag > self.data_height - 0.5:
                continue
            # Cut along left boundary.
            if e.measurement_qubit.real * 3 < e.measurement_qubit.imag - 1:
                # Handle corner case.
                if e.measurement_qubit.imag < self.data_height - 2:
                    continue
            # Cuts along right boundary.
            if (e.measurement_qubit.real - self.data_width) * 3 > e.measurement_qubit.imag:
                continue
            kept_elements.append(e)
        elements = kept_elements

        # Trim leaves.
        hexes = HoneycombLayout.hexes_given_edge_plan_elements(elements)
        data_qubits_in_hex = {
            q for h in hexes for q in h.data_qubits
        }
        remaining_elements = []
        for e in elements:
            kept_data_qubits = tuple(q if q in data_qubits_in_hex else None for q in e.data_qubit_order)
            if any(q is not None for q in kept_data_qubits):
                remaining_elements.append(StabilizerPlanElement(
                    bases=e.bases,
                    measurement_qubit=e.measurement_qubit,
                    data_qubit_order=kept_data_qubits,
                ))
        elements = remaining_elements

        return StabilizerPlan(tuple(sorted(
            elements,
            key=lambda e: complex_key(e.measurement_qubit),
        )))

    @staticmethod
    def hexes_given_edge_plan_elements(edge_plan_elements: Iterable[StabilizerPlanElement]):
        data_qubits = {q for e in edge_plan_elements for q in e.data_coords_set()}
        measurement_qubits = {e.measurement_qubit for e in edge_plan_elements}
        hexes: List[HoneycombHex] = []
        for q in data_qubits:
            if all(q + d in measurement_qubits for d in HEX_MEASURE_OFFSETS):
                hexes.append(HoneycombHex.from_top_left(top_left=q))

        return tuple(sorted(
            hexes,
            key=lambda e: complex_key(e.top_left),
        ))

    @functools.cached_property
    def data_qubit_set(self) -> FrozenSet[complex]:
        return frozenset(q
                         for e in self.edge_plan.elements
                         for q in e.data_qubit_order
                         if q is not None)

    @functools.cached_property
    def vertical_observable_path(self) -> Tuple[complex, ...]:
        x = (self.data_height // 6) * 2 - 1 + (self.data_height % 2)
        result = sorted([q for q in self.data_qubit_set if q.real == x], key=complex_key)
        result.append(result[-1] + 1j)
        result.insert(0, result[0] - 1j)
        return tuple(result)

    @functools.cached_property
    def horizontal_observable_path(self) -> Tuple[complex, ...]:
        result = sorted(
            [q for q in self.data_qubit_set if q.imag in [1, 2]],
            key=lambda q: (q.real, (q.imag + q.real) % 2 == 0)
        )
        result.append(result[-1] + 1)
        result.insert(0, result[0] - 1)
        return tuple(result)

    @functools.cached_property
    def vertical_observable_measurement_qubit_set(self) -> FrozenSet[complex]:
        p = self.vertical_observable_path
        return frozenset((p[k] + p[k + 1]) / 2 for k in range(len(p) - 1))

    @functools.cached_property
    def horizontal_observable_measurement_qubit_set(self) -> FrozenSet[complex]:
        p = self.horizontal_observable_path
        return frozenset((p[k] + p[k + 1]) / 2 for k in range(len(p) - 1))

    def vertical_observable(self, step: int) -> StabilizerPlanElement:
        data_qubits = self.vertical_observable_path
        steps = [
            'X_X',
            'XX_',
            'ZZ_',
            '_ZZ',
            '_YY',
        ]
        bases = (steps[step] * len(data_qubits))[:len(data_qubits)]
        return StabilizerPlanElement(
            bases=tuple('_' + bases[:-2] + '_'),
            data_qubit_order=data_qubits,
            measurement_qubit=data_qubits[0]
        )

    def horizontal_observable(self, step: int) -> StabilizerPlanElement:
        data_qubits = self.horizontal_observable_path
        steps = [
            'ZZZZ',
            'YYYY',
            '_YY_',
            '_XX_',
            'X__X',
        ]
        bases = (steps[step] * len(data_qubits))[:len(data_qubits)]
        return StabilizerPlanElement(
            bases=tuple('_' + bases[:-2] + '_'),
            data_qubit_order=data_qubits,
            measurement_qubit=data_qubits[0]
        )

    def observable_plan(self, step: int) -> StabilizerPlan:
        return StabilizerPlan(
            elements=(),
            observables=(self.horizontal_observable(step), self.vertical_observable(step)),
        )

    @functools.cached_property
    def hexes(self) -> Tuple[HoneycombHex, ...]:
        return HoneycombLayout.hexes_given_edge_plan_elements(self.edge_plan.elements)

    @functools.cached_property
    def all_qubits_set(self) -> FrozenSet[complex]:
        return frozenset(
            q
            for e in self.edge_plan.elements
            for q in e.used_coords_set()
        )

    @functools.cached_property
    def boundary_hex_set(self) -> FrozenSet[HoneycombHex]:
        expanded_hexes = {
            HoneycombHex.from_top_left(top_left=q + x + y, present_qubits=self.all_qubits_set)
            for q in self.data_qubit_set
            for x in [0, -1, -2]
            for y in [0, -1j, -2j]
            if ((q + x + y).real + (q + x + y).imag) % 2 == 0
        }
        bulk_hexes = set(self.hexes)
        return frozenset(
            h
            for h in expanded_hexes
            if h.measurement_qubits and h not in bulk_hexes and len(h.data_qubits) != 3
        )

    @functools.cached_property
    def hex_plan(self) -> StabilizerPlan:
        return StabilizerPlan(tuple(
            e.as_plan_element()
            for e in self.hexes + tuple(self.boundary_hex_set)
        ))

    @functools.cached_property
    def x_edges(self) -> StabilizerPlan:
        return StabilizerPlan(tuple(e for e in self.edge_plan.elements if e.bases[0] == 'X'))

    @functools.cached_property
    def y_edges(self) -> StabilizerPlan:
        return StabilizerPlan(tuple(e for e in self.edge_plan.elements if e.bases[0] == 'Y'))

    @functools.cached_property
    def z_edges(self) -> StabilizerPlan:
        return StabilizerPlan(tuple(e for e in self.edge_plan.elements if e.bases[0] == 'Z'))

    @functools.cached_property
    def x_measurement_qubits(self) -> FrozenSet[complex]:
        return frozenset(e.measurement_qubit for e in self.x_edges.elements)

    @functools.cached_property
    def y_measurement_qubits(self) -> FrozenSet[complex]:
        return frozenset(e.measurement_qubit for e in self.y_edges.elements)

    @functools.cached_property
    def z_measurement_qubits(self) -> FrozenSet[complex]:
        return frozenset(e.measurement_qubit for e in self.z_edges.elements)

    def xyz_edges(self, basis: str) -> StabilizerPlan:
        return {
            'X': self.x_edges,
            'Y': self.y_edges,
            'Z': self.z_edges,
        }[basis]

    def xyz_measurement_qubits(self, basis: str) -> FrozenSet[complex]:
        return {
            'X': self.x_measurement_qubits,
            'Y': self.y_measurement_qubits,
            'Z': self.z_measurement_qubits,
        }[basis]

    @functools.cached_property
    def x_hex(self) -> Tuple[HoneycombHex, ...]:
        return tuple(e for e in self.hexes if e.basis == 'X')

    @functools.cached_property
    def y_hex(self) -> Tuple[HoneycombHex, ...]:
        return tuple(e for e in self.hexes if e.basis == 'Y')

    @functools.cached_property
    def z_hex(self) -> Tuple[HoneycombHex, ...]:
        return tuple(e for e in self.hexes if e.basis == 'Z')

    def xyz_hex(self, basis: str) -> Tuple[HoneycombHex, ...]:
        return {
            'X': self.x_hex,
            'Y': self.y_hex,
            'Z': self.z_hex,
        }[basis]

    def circuit(self) -> stim.Circuit:
        circuit = stim.Circuit()
        q2i = {q: i for i, q in enumerate(sorted(self.edge_plan.used_coords_set(), key=complex_key))}
        epr_ancilla = -2
        assert epr_ancilla not in q2i
        q2i[epr_ancilla] = len(q2i)
        for q, i in q2i.items():
            circuit.append("QUBIT_COORDS", i, [q.real, q.imag])

        target_xyz = {
            'X': stim.target_x,
            'Y': stim.target_y,
            'Z': stim.target_z,
        }
        tracker = MeasurementTracker()
        measured_bases = ''

        def append_obs_measurement(obs: StabilizerPlanElement, index: int):
            targets = []
            for p, q in zip(obs.bases, obs.data_qubit_order):
                assert p in '_XYZ'
                if p != '_':
                    targets.append(target_xyz[p](q2i[q]))
                    targets.append(stim.target_combiner())
            targets.append(target_xyz['XZ'[index]](q2i[epr_ancilla]))
            circuit.append('MPP', targets)
            circuit.append('OBSERVABLE_INCLUDE', stim.target_rec(-1), index)
            circuit.append("TICK")

        use_vertical_obs = True
        use_horizontal_obs = True
        if use_vertical_obs:
            append_obs_measurement(self.vertical_observable(1), 0)
        if use_horizontal_obs:
            append_obs_measurement(self.horizontal_observable(1), 1)

        for r in range(self.rounds):
            for edge_basis in EDGE_MEASUREMENT_SEQUENCE:
                noise = self.noise if r != 0 and r != self.rounds - 1 else 0
                target = target_xyz[edge_basis]
                added_measurements = []
                for edge in self.xyz_edges(edge_basis).elements:
                    edge: StabilizerPlanElement
                    data_coords = edge.data_coords_set()
                    targets = []
                    ts = []
                    for q in sorted(data_coords, key=complex_key):
                        targets.append(target(q2i[q]))
                        ts.append(q2i[q])
                        targets.append(stim.target_combiner())
                    targets.pop()
                    circuit.append(f"DEPOLARIZE{len(ts)}", ts, noise)
                    circuit.append("MPP", targets, noise)
                    added_measurements.append(edge.measurement_qubit)
                tracker.add_measurements(*added_measurements)
                if edge_basis != 'X':
                    if use_vertical_obs:
                        circuit.append(
                            "OBSERVABLE_INCLUDE",
                            tracker.get_record_targets(*(set(added_measurements) & self.vertical_observable_measurement_qubit_set)),
                            0)
                    if use_horizontal_obs:
                        circuit.append(
                            "OBSERVABLE_INCLUDE",
                            tracker.get_record_targets(*(set(added_measurements) & self.horizontal_observable_measurement_qubit_set)),
                            1)

                measured_bases += edge_basis
                cmp = find_hex_comparison(measured_bases)
                if cmp is not None:
                    hex_basis, comparisons = cmp
                    for h in self.xyz_hex(hex_basis):
                        measurements = []
                        for c in comparisons:
                            for m in h.measurement_qubits:
                                if m in self.xyz_measurement_qubits(c.v):
                                    measurements.append(Prev(m, offset=c.offset))
                        tracker.append_detector(
                            *measurements,
                            coords=[h.center.real, h.center.imag, 0],
                            out_circuit=circuit,
                        )
                m2e = self.measure_to_element_dict

                if len(measured_bases) >= 3 and measured_bases[-3] == measured_bases[-1]:
                    for h in self.boundary_hex_set:
                        measured_leaves = [m for m in h.measurement_qubits if m2e[m].is_leaf() and m2e[m].common_basis() == measured_bases[-1]]
                        if len(h.data_qubits) == 2 and len(measured_leaves) == 2 and h.basis != 'X':
                            a, b = measured_leaves
                            tracker.append_detector(
                                a, Prev(a),
                                b, Prev(b),
                                coords=[h.center.real, h.center.imag, 0],
                                out_circuit=circuit,
                            )

                        if len(h.data_qubits) == 4 and len(measured_leaves) == 2 and h.basis != 'X':
                            kept = [q for q in h.measurement_qubits if m2e[q].common_basis() != 'X']
                            tracker.append_detector(
                                *[q for q in kept],
                                *[Prev(q) for q in kept],
                                coords=[h.center.real, h.center.imag, 0],
                                out_circuit=circuit,
                            )

                if len(measured_bases) >= 5 and measured_bases[-5] == measured_bases[-1]:
                    for h in self.boundary_hex_set:
                        measured_leaves = [m for m in h.measurement_qubits if m2e[m].is_leaf() and m2e[m].common_basis() == measured_bases[-1]]
                        if len(h.data_qubits) == 2 and len(measured_leaves) == 2 and h.basis == 'X':
                            tracker.append_detector(
                                *h.measurement_qubits,
                                *[Prev(q) for q in h.measurement_qubits],
                                coords=[h.center.real, h.center.imag, 0],
                                out_circuit=circuit,
                            )

                        if len(h.data_qubits) == 4 and len(measured_leaves) == 2 and h.basis == 'X':
                            kept = [q for q in h.measurement_qubits]
                            tracker.append_detector(
                                *[q for q in kept],
                                *[Prev(q) for q in kept],
                                coords=[h.center.real, h.center.imag, 0],
                                out_circuit=circuit,
                            )

                circuit.append("SHIFT_COORDS", [], [0, 0, 1])


                circuit.append("TICK")

        if use_vertical_obs:
            append_obs_measurement(self.vertical_observable(1), 0)
        if use_horizontal_obs:
            append_obs_measurement(self.horizontal_observable(1), 1)


        return circuit

    def to_decoding_problem(self, decoder: str) -> DecodingProblem:
        return DecodingProblem(
            circuit_maker=self.circuit,
            desc=DecodingProblemDesc(
                data_width=self.data_width,
                data_height=self.data_height,
                code_distance=min(self.data_width - 2, self.data_height // 3 * 2 - 2),
                num_qubits=len(self.all_qubits_set),
                rounds=self.rounds,
                noise=self.noise,
                circuit_style=f"bounded_honeycomb_memory",
                preserved_observable="EPR",
                decoder=decoder,
            )
        )


def main():
    out_dir = pathlib.Path(__file__).parent.parent.parent.parent.parent / 'out'
    layout = HoneycombLayout.from_code_distance(distance=10,
                                                rounds=10,
                                                noise=0.001)
    edge_plan = layout.edge_plan
    hex_plan = layout.hex_plan
    plans = []
    for step in range(5):
        observable_plan = layout.observable_plan(step)
        plans.append(StabilizerPlan(
            elements=tuple([*hex_plan.elements, *edge_plan.elements]),
            observables=observable_plan.observables))
    with open(out_dir / 'tmp.svg', 'w') as f:
        print(StabilizerPlan.svg(*plans, show_order=False), file=f)
    circuit = layout.circuit()
    with open(out_dir / 'tmp.html', 'w') as f:
        print(stim_circuit_html_viewer(circuit=circuit, width=500, height=500), file=f)
    error_model: stim.DetectorErrorModel = circuit.detector_error_model(decompose_errors=True)

    shortest_error = error_model.shortest_graphlike_error()
    print(f"graphlike code distance = {len(shortest_error)}")
    for e in shortest_error:
        print("    ", e)
    #
    # shortest_error_circuit = circuit.shortest_graphlike_error(ignore_ungraphlike_errors=True)
    # print("Circuit equivalents")
    # for e in shortest_error_circuit:
    #     print("    " + e.replace('\n', '\n    '))
    # print(f"graphlike code distance = {len(shortest_error)}")
    #
    # assert circuit.detector_error_model(decompose_errors=True) is not None


if __name__ == '__main__':
    main()
