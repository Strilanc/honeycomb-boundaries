import collections
import dataclasses
import functools
import pathlib
from typing import List, Tuple, Iterable, FrozenSet, Optional, Dict

import stim

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
EDGE_BASIS_SEQUENCE = 'XZY'
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


@dataclasses.dataclass
class HoneycombHex:
    top_left: complex
    basis: str

    @property
    def center(self) -> complex:
        return self.top_left + 0.5 + 1j

    def measurement_qubits(self) -> Tuple[complex, ...]:
        return tuple(self.top_left + d for d in HEX_MEASURE_OFFSETS)

    def data_qubits(self) -> Tuple[complex, ...]:
        return tuple(self.top_left + d for d in HEX_DATA_OFFSETS)

    def as_plan_element(self) -> StabilizerPlanElement:
        return StabilizerPlanElement(
            bases=(self.basis,) * 6,
            data_qubit_order=self.data_qubits(),
            measurement_qubit=self.measurement_qubits()[0],
        )


class HoneycombLayout:
    def __init__(self, *, data_width: int, data_height: int):
        self.data_width = data_width
        self.data_height = data_height

    @functools.cached_property
    def boundary_pairs(self) -> Tuple[Tuple[StabilizerPlanElement, complex, complex, str], ...]:
        result: List[Tuple[StabilizerPlanElement, complex, complex, str]] = []

        d = self.leaf_data_to_leaf_element_dict
        for e in self.edge_plan.elements:
            if all(q in d for q in e.data_qubit_order):
                a, b = e.data_qubit_order
                da = d[a]
                db = d[b]
                basis = da.common_basis()
                assert db.common_basis() == basis and basis is not None
                result.append((e, da.measurement_qubit, db.measurement_qubit, basis))
        return tuple(sorted(result, key=lambda e: complex_key(e[0].measurement_qubit)))

    def leaf_neighbor_of(self, edge: StabilizerPlanElement) -> Optional[StabilizerPlanElement]:
        results = []
        d = self.leaf_data_to_leaf_element_dict
        for q in edge.data_coords_set():
            if q in d:
                results.append(d[q])
        assert len(results) <= 1
        if results:
            return results[0]
        return None

    def leaf_neighbor_neighbor_of(self, q: complex) -> Optional[Tuple[StabilizerPlanElement, StabilizerPlanElement]]:
        results: List[Tuple[StabilizerPlanElement, StabilizerPlanElement]] = []
        for e2 in self.data_to_elements_dict[q]:
            e3 = self.leaf_neighbor_of(e2)
            if e3 is not None:
                results.append((e2, e3))
        assert len(results) <= 1
        if results:
            return results[0]
        return None

    @functools.cached_property
    def boundary_squares(self) -> Tuple[Tuple[StabilizerPlanElement, Tuple[complex, complex], Tuple[complex, complex], str], ...]:
        result: List[Tuple[StabilizerPlanElement, Tuple[complex, complex], Tuple[complex, complex], str]] = []

        data_hex_counter = collections.Counter(
            q
            for h in self.hexes
            for q in h.data_qubits()
        )
        for e in self.edge_plan.elements:
            if len(e.data_coords_set()) == 2 and all(data_hex_counter[q] == 2 for q in e.data_coords_set()):
                a, b = e.data_coords_set()
                a1, a2 = self.leaf_neighbor_neighbor_of(a)
                b1, b2 = self.leaf_neighbor_neighbor_of(b)
                result.append((e, (a1.measurement_qubit, b1.measurement_qubit), (a2.measurement_qubit, b2.measurement_qubit), e.common_basis()))
        return tuple(sorted(result, key=lambda e: complex_key(e[0].measurement_qubit)))

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
    def measure_to_elements_dict(self) -> Dict[complex, StabilizerPlanElement]:
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
        for x in range(self.data_width):
            for y in range(self.data_height):
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
            # Cuts along top boundary.
            if e.measurement_qubit.imag <= 1:
                continue
            # Cuts along bottom boundary.
            if e.measurement_qubit.imag >= self.data_height - 2:
                continue
            # Cuts along left boundary.
            if e.measurement_qubit.imag + e.measurement_qubit.real * 3 <= 17:
                continue
            # Cuts along right boundary.
            if e.measurement_qubit.imag + e.measurement_qubit.real * 3 >= 54:
                # Handle corner case.
                if e.measurement_qubit.imag < self.data_height - 4:
                    continue
            kept_elements.append(e)
        elements = kept_elements

        # Trim leaves.
        hexes = HoneycombLayout.hexes_given_edge_plan_elements(elements)
        data_qubits_in_hex = {
            q for h in hexes for q in h.data_qubits()
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
            basis = EDGE_BASIS_SEQUENCE[(2 + int(q.imag)) % 3]
            if all(q + d in measurement_qubits for d in HEX_MEASURE_OFFSETS):
                hexes.append(HoneycombHex(top_left=q, basis=basis))

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
        result = sorted([q for q in self.data_qubit_set if q.real == 10], key=complex_key)
        result.append(result[-1] + 1j)
        result.insert(0, result[0] - 1j)
        return tuple(result)

    @functools.cached_property
    def horizontal_observable_path(self) -> Tuple[complex, ...]:
        result = sorted(
            [q for q in self.data_qubit_set if q.imag in [6, 7]],
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
    def hex_plan(self) -> StabilizerPlan:
        return StabilizerPlan(tuple(
            e.as_plan_element()
            for e in self.hexes
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

        for r in range(10):
            for edge_basis in EDGE_MEASUREMENT_SEQUENCE:
                if r in [4, 5]:
                    circuit.append("DEPOLARIZE1", [q2i[q] for q in self.data_qubit_set], 0.001)
                target = target_xyz[edge_basis]
                added_measurements = []
                for edge in self.xyz_edges(edge_basis).elements:
                    edge: StabilizerPlanElement
                    data_coords = edge.data_coords_set()
                    targets = []
                    for q in sorted(data_coords, key=complex_key):
                        targets.append(target(q2i[q]))
                        targets.append(stim.target_combiner())
                    targets.pop()
                    circuit.append("MPP", targets)
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
                            for m in h.measurement_qubits():
                                if m in self.xyz_measurement_qubits(c.v):
                                    measurements.append(Prev(m, offset=c.offset))
                        tracker.append_detector(
                            *measurements,
                            coords=[h.center.real, h.center.imag, 0],
                            out_circuit=circuit,
                        )
                if len(measured_bases) >= 3 and measured_bases[-3] == measured_bases[-1]:
                    for boundary, a, b, basis in self.boundary_pairs:
                        if boundary.common_basis() == 'X' and basis == measured_bases[-1]:
                            tracker.append_detector(
                                a, Prev(a),
                                b, Prev(b),
                                coords=[boundary.measurement_qubit.real, boundary.measurement_qubit.imag, 0],
                                out_circuit=circuit,
                            )
                    for boundary, (a1, b1), (a2, b2), basis in self.boundary_squares:
                        if boundary.common_basis() == measured_bases[-1] and self.measure_to_elements_dict[
                            a1].common_basis() == 'X':
                            tracker.append_detector(
                                a2, Prev(a2),
                                b2, Prev(b2),
                                boundary.measurement_qubit, Prev(boundary.measurement_qubit),
                                coords=[boundary.measurement_qubit.real, boundary.measurement_qubit.imag, 0],
                                out_circuit=circuit,
                            )
                if len(measured_bases) >= 5 and measured_bases[-5] == measured_bases[-1]:
                    for boundary, a, b, basis in self.boundary_pairs:
                        if boundary.common_basis() != 'X' and basis == measured_bases[-1]:
                            tracker.append_detector(
                                a, Prev(a),
                                b, Prev(b),
                                boundary.measurement_qubit,
                                Prev(boundary.measurement_qubit),
                                coords=[boundary.measurement_qubit.real, boundary.measurement_qubit.imag, 0],
                                out_circuit=circuit,
                            )
                    for boundary, (a1, b1), (a2, b2), basis in self.boundary_squares:
                        if boundary.common_basis() == measured_bases[-1] and self.measure_to_elements_dict[
                            a1].common_basis() == measured_bases[-2]:
                            tracker.append_detector(
                                a1, Prev(a1),
                                a2, Prev(a2),
                                b1, Prev(b1),
                                b2, Prev(b2),
                                boundary.measurement_qubit, Prev(boundary.measurement_qubit),
                                coords=[boundary.measurement_qubit.real, boundary.measurement_qubit.imag, 0],
                                out_circuit=circuit,
                            )
                circuit.append("SHIFT_COORDS", [], [0, 0, 1])


                circuit.append("TICK")

        if use_vertical_obs:
            append_obs_measurement(self.vertical_observable(1), 0)
        if use_horizontal_obs:
            append_obs_measurement(self.horizontal_observable(1), 1)


        return circuit


def main():
    out_dir = pathlib.Path(__file__).parent.parent.parent.parent.parent / 'out'
    layout = HoneycombLayout(data_width=24, data_height=16)
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
    error_model: stim.DetectorErrorModel = circuit.detector_error_model()

    shortest_error = error_model.shortest_graphlike_error(ignore_ungraphlike_errors=True)
    print(f"graphlike code distance = {len(shortest_error)}")
    for e in shortest_error:
        print("    ", e)

    shortest_error_circuit = circuit.shortest_graphlike_error(ignore_ungraphlike_errors=True)
    print("Circuit equivalents")
    for e in shortest_error_circuit:
        print("    " + e.replace('\n', '\n    '))
    print(f"graphlike code distance = {len(shortest_error)}")

    assert circuit.detector_error_model(decompose_errors=True) is not None


if __name__ == '__main__':
    main()
