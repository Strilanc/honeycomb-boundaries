import collections
import dataclasses
import functools
import pathlib
from typing import List, Tuple, Iterable, FrozenSet

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
            [q for q in self.data_qubit_set if q.imag in [4, 5]],
            key=lambda q: (q.real, (q.imag + q.real) % 2 == 0)
        )
        result.append(result[-1] + 1)
        result.insert(0, result[0] - 1)
        return tuple(result)

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

    def xyz_edges(self, basis: str) -> StabilizerPlan:
        return {
            'X': self.x_edges,
            'Y': self.y_edges,
            'Z': self.z_edges,
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
        for q, i in q2i.items():
            circuit.append_operation("QUBIT_COORDS", [i], [q.real, q.imag])

        target_xyz = {
            'X': stim.target_x,
            'Y': stim.target_y,
            'Z': stim.target_z,
        }
        tracker = MeasurementTracker()
        for r in range(10):
            for k in range(3):
                edge_basis = EDGE_BASIS_SEQUENCE[k]
                target = target_xyz[edge_basis]
                for edge in self.xyz_edges(edge_basis).elements:
                    edge: StabilizerPlanElement
                    data_coords = edge.data_coords_set()
                    targets = []
                    for q in sorted(data_coords, key=complex_key):
                        targets.append(target(q2i[q]))
                        targets.append(stim.target_combiner())
                    targets.pop()
                    circuit.append_operation("MPP", targets)
                    tracker.add_measurements(edge.measurement_qubit)

                if r > 3:
                    hex_basis = EDGE_BASIS_SEQUENCE[k - 2]
                    if edge_basis == 'Y' and hex_basis == 'Z':
                        for hex in self.xyz_hex(hex_basis):
                            tracker.append_detector(
                                *hex.measurement_qubits(),
                                *[Prev(e) for e in hex.measurement_qubits()],
                                coords=[hex.center.real, hex.center.imag],
                                out_circuit = circuit,
                            )

                circuit.append_operation("TICK")
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
    assert circuit.detector_error_model(decompose_errors=True) is not None


if __name__ == '__main__':
    main()
