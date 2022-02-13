import collections
import dataclasses
import functools
from typing import List, Tuple, Iterable, FrozenSet, Optional, Dict, AbstractSet

import stim

from hcb.codes.surface.stabilizer_plan_problem import StabilizerPlanProblem
from hcb.tools.analysis.collecting import DecodingProblem, DecodingProblemDesc
from hcb.tools.gen.circuit_canvas import complex_key
from hcb.tools.gen.measurement_tracker import MeasurementTracker, Prev
from hcb.tools.gen.stabilizer_plan import StabilizerPlan, StabilizerPlanElement

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
SI1000_DATA_ROTATION_SEQUENCE: Tuple[Tuple[str, str, str, str], Tuple[str, str, str, str]] = (
    (
        'C_ZYX',
        'C_ZYX',
        'C_ZYX',
        'I',
    ),
    (
        'C_ZYX',
        'C_XYZ',
        'C_XYZ',
        'C_ZYX',
    ),
)


@dataclasses.dataclass
class ComparisonRule:
    kind: str
    filter_basis: str
    last_measure_basis: str
    product: Tuple[Prev, ...]


INIT_COMPARISON_RULES: Dict[Tuple[Optional[str], str, Optional[str]], ComparisonRule] = {
    ('X', 'X', None): ComparisonRule(
        kind='edge',
        filter_basis='X',
        product=(Prev('X', 0),),
        last_measure_basis='X',
    ),
    ('X', 'XYZ', None): ComparisonRule(
        kind='hex',
        filter_basis='X',
        product=(Prev('Y', 0), Prev('Z', 0)),
        last_measure_basis='Z',
    ),
    ('Y', 'XY', None): ComparisonRule(
        kind='hex',
        filter_basis='Z',
        product=(Prev('Y', 0),),
        last_measure_basis='Y',
    ),
    ('Y', 'XYZX', None): ComparisonRule(
        kind='hex',
        filter_basis='Y',
        product=(Prev('X', 0), Prev('Z', 0)),
        last_measure_basis='X',
    ),
}

def checkerboard_type(a: complex) -> bool:
    return (a.real + a.imag) % 2 == 1


def data_checkerboard_sort(a: complex, b: complex) -> Tuple[complex, complex]:
    a0 = checkerboard_type(a)
    b0 = checkerboard_type(b)
    assert a0 != b0
    if a0 > b0:
        a, b = b, a
    return a, b


OSCILLATING_BOUNDARY_OFFSETS = {
    0: -0.25,
    0.5: -0.25,
    1: +0.25,
    1.5: +0.25,
    2: +0.25,
    2.5: +0.25,
    3: +0.25,
    3.5: -0.25,
    4: -0.25,
    4.5: -0.25,
    5: -0.75,
    5.5: -0.25,
}

def comparisons_for_step(*,
                         previous_measurements: str,
                         data_init_basis: Optional[str],
                         data_measure_basis: Optional[str]) -> Optional[ComparisonRule]:
    """Determines sets of edge measurements to compare to get detectors at various times.

    Args:
        previous_measurements: The basis of each layer of edge measurements performed so far.
        data_init_basis: The basis that data qubits of the patch were initially reset into.
        data_measure_basis: The basis that data qubits of the patch were just measured in.
            (Use None until the actual data measurement occurs.)

    Returns:
        A (stabilizer_basis, detector_parts) tuple.
            stabilizer_basis: The color of hex this applies to. For example, after measuring X edges
                then Y edges the basis will be 'Z' because those are the stabilizers that have been
                formed again and can be compared to previous values.
            detector_parts: A tuple of `Prev(edge_basis, lookback)` values, indicating which edges
                at which time (around the border of each of the stabilizers) to multiply together
                to get a deterministic result. Should include the current edge measurement.
    """
    # Special case timelike boundaries.
    key = (data_init_basis, previous_measurements, data_measure_basis)
    if key in INIT_COMPARISON_RULES:
        return INIT_COMPARISON_RULES[key]

    if len(previous_measurements) < 2:
        return None
    def piece(pos: int) -> Prev:
        char = previous_measurements[pos]
        return Prev(char, offset=sum(c == char for c in previous_measurements[pos + 1:]))

    # Look at current measurement pair.
    match = set(previous_measurements[-2:])
    remainder, = set('XYZ') - match

    # Find the previous time this pair occurred next to each other.
    n = len(previous_measurements)
    for k in range(n - 2)[::-1]:
        if set(previous_measurements[k:k+2]) == match:
            pieces = set()
            for p in [piece(n-1), piece(n-2), piece(k), piece(k+1)]:
                pieces ^= {p}
            return ComparisonRule(
                kind='hex',
                filter_basis=remainder,
                product=tuple(sorted(pieces, key=lambda pc: (pc.v, pc.offset))),
                last_measure_basis=previous_measurements[-1],
            )

    return None


@dataclasses.dataclass(unsafe_hash=True, frozen=True)
class HoneycombHex:
    top_left: complex
    data_qubits: Tuple[complex, ...]
    measurement_qubits: Tuple[complex, ...]

    def __post_init__(self):
        assert (self.top_left.real + self.top_left.imag) % 2 == 0

    def leaf_basis(self, m2e: Dict[complex, StabilizerPlanElement]) -> Optional[str]:
        if len(self.data_qubits) == 6:
            return None
        result, = {m2e[m].common_basis() for m in self.measurement_qubits if m2e[m].is_leaf()}
        return result

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
    def __init__(self,
                 *,
                 data_width: int,
                 data_height: int,
                 rounds: int,
                 noise_level: float,
                 noisy_gate_set: str,
                 tested_observable: str,
                 sheared: bool = False):
        """
        Args:
            data_width: The left-to-right diameter of the patch.
                Note that the patch is sheared. It has the same width at the top and at the bottom,
                but the bottom is further to the right than the top.
            data_height: The top-to-bottom diameter of the patch.
            rounds: The number of times to measure each measurement qubits.
                One round measures each edge once.
                Must be at least 4 and even.
                Note that rounds alternate between measuring X-then-Y-then-Z and X-then-Z-then-Y.
            noise_level: The strength of noise added into circuit.
            noisy_gate_set: The type of noisy gates available.
            tested_observable: The observable to initialize and later measure. Valid values are:
                "V": Fault-tolerantly initialize, protect, and measure the vertical observable.
                "H": Fault-tolerantly initialize, protect, and measure the horizontal observable.
                "EPR": Magically noiselessly initialize the logical qubit into a Bell pair entangled
                    with a noiseless ancilla, protect both the horizontal and vertical logical
                    observables against noise, then magically noiselessly perform a Bell basis
                    measurement.

                    This mode is useful for verifying that the two observables actually form a
                    logical qubit, since if didn't anti-commute it would be impossible to correlate
                    them both with the ancilla qubit's anti-commuting X and Z observables.
        """
        self.tested_observable = tested_observable
        self.data_width = data_width
        self.data_height = data_height
        self.rounds = rounds
        self.noise_level = noise_level
        self.noisy_gate_set = noisy_gate_set
        self.sheared = sheared
        assert self.tested_observable in ['V', 'H', 'EPR']
        assert self.data_width % 2 == 0 or not self.sheared
        assert self.data_height % 3 == 0
        assert self.data_width * 3 >= self.data_height or not self.sheared
        assert self.rounds % 2 == 0 and self.rounds >= 4
        assert self.data_width > 0
        assert self.data_height > 0


    @staticmethod
    def unsheared_size_for_code_distance(distance: int,
                                         gate_set: str) -> Tuple[int, int]:
        if gate_set in ['EM3_v1', 'EM3_v2']:
            return distance * 2, distance * 3
        if gate_set in ['SD6', 'SI1000']:
            w = distance + 1
            h = distance * 2
            while h % 3:
                h += 1
            assert h % 3 == 0, str(h)
            return w, h
        raise NotImplementedError()

    def horizontal_graphlike_code_distance(self) -> int:
        if self.noisy_gate_set in ['EM3_v1', 'EM3_v2']:
            return self.data_width // 2
        if self.noisy_gate_set in ['SD6', 'SI1000']:
            return self.data_width - 1
        raise NotImplementedError(self.noisy_gate_set)

    def vertical_graphlike_code_distance(self) -> int:
        if self.noisy_gate_set in ['EM3_v1', 'EM3_v2']:
            return self.data_height // 3
        if self.noisy_gate_set in ['SD6', 'SI1000']:
            return self.data_height // 2
        raise NotImplementedError()

    def graphlike_code_distance(self) -> int:
        return min(self.horizontal_graphlike_code_distance(),
                   self.vertical_graphlike_code_distance())

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
        elements: List[StabilizerPlanElement] = []
        for x in range(-5, self.data_width * 2 + 2):
            for y in range(-5, self.data_height + 2):
                # Add vertically oriented edges.
                c = x + 1j * y
                h_basis = EDGE_BASIS_SEQUENCE[int(y) % 3]
                elements.append(StabilizerPlanElement(
                    bases=(h_basis,) * 2,
                    measurement_qubit=c + 0.5j,
                    data_qubit_order=data_checkerboard_sort(
                        c,
                        c + 1j,
                    ),
                ))

                # Add horizontally oriented edges.
                if (x + y) & 1 == 0:
                    v_basis = EDGE_BASIS_SEQUENCE[int(y) % 3 - 2]
                    elements.append(StabilizerPlanElement(
                        bases=(v_basis,) * 2,
                        measurement_qubit=c + 0.5,
                        data_qubit_order=data_checkerboard_sort(
                            c,
                            c + 1,
                        ),
                    ))

        # Cut edges to form boundaries.
        kept_elements = []
        for e in elements:
            # Cut along top boundary.
            if e.measurement_qubit.imag < -0.5:
                continue
            # Cut along bottom boundary.
            if e.measurement_qubit.imag > self.data_height - 0.5:
                continue

            if self.sheared:
                # Cut along left boundary.
                if e.measurement_qubit.real * 3 < e.measurement_qubit.imag - 1:
                    continue
                # Cut along right boundary.
                if (e.measurement_qubit.real - self.data_width) * 3 > e.measurement_qubit.imag:
                    continue
            else:
                # Cut along left boundary.
                if e.measurement_qubit.real < OSCILLATING_BOUNDARY_OFFSETS[e.measurement_qubit.imag % 6]:
                    continue
                # Cut along right boundary.
                dy = 0 if self.data_width % 2 == 1 else 3
                if e.measurement_qubit.real > self.data_width - OSCILLATING_BOUNDARY_OFFSETS[(dy + e.measurement_qubit.imag) % 6]:
                    continue
            kept_elements.append(e)
        elements = kept_elements

        data_qubit_measure_count = collections.Counter(
            q
            for e in elements
            for q in e.data_qubit_order
        )
        truncated_elements = []
        for e in elements:
            dqs = tuple(
                q if data_qubit_measure_count[q] == 3 else None
                for q in e.data_qubit_order
            )
            assert any(e is not None for e in dqs)
            truncated_elements.append(StabilizerPlanElement(
                bases=e.bases,
                measurement_qubit=e.measurement_qubit,
                data_qubit_order=dqs,
            ))
        elements = truncated_elements

        return StabilizerPlan(tuple(sorted(
            elements,
            key=lambda edge: complex_key(edge.measurement_qubit),
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
        if self.sheared:
            x = (self.data_height // 6) * 2 + (self.data_height % 2)
        else:
            x = self.data_width // 2
        result = sorted([q for q in self.data_qubit_set if q.real == x], key=complex_key)
        result.append(result[-1] + 1j)
        result.insert(0, result[0] - 1j)
        return tuple(result)

    @functools.cached_property
    def horizontal_observable_path(self) -> Tuple[complex, ...]:
        y = (self.data_height // 6) * 3 + 1
        result = sorted(
            [q for q in self.data_qubit_set if q.imag in [y, y + 1]],
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

    def vertical_observable(self, layout_index: int) -> StabilizerPlanElement:
        data_qubits = self.vertical_observable_path
        steps = [
            'X_X',
            'XX_',
            'ZZ_',
            '_ZZ',
            '_YY',
        ]
        bases = (steps[layout_index] * len(data_qubits))[:len(data_qubits)]
        return StabilizerPlanElement(
            bases=tuple('_' + bases[:-2] + '_'),
            data_qubit_order=data_qubits,
            measurement_qubit=data_qubits[0]
        )

    def horizontal_observable(self, layout_index: int) -> StabilizerPlanElement:
        data_qubits = self.horizontal_observable_path
        steps = [
            'ZZZZ',
            'YYYY',
            '_YY_',
            '_XX_',
            'X__X',
        ]
        bases = (steps[layout_index] * len(data_qubits))[:len(data_qubits)]
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
        result = frozenset(
            h
            for h in expanded_hexes
            if h.measurement_qubits
            and h not in bulk_hexes
            and len(h.data_qubits) in [2, 4]
        )

        # Sanity check.
        m2e = self.measure_to_element_dict
        for h in result:
            assert len(h.data_qubits) in [2, 4]
            assert h.leaf_basis(m2e=m2e) in ['Y', 'Z']

        return result

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

    def xyz_measurement_qubit_set(self, basis: str) -> FrozenSet[complex]:
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

    def append_hex_based_detector(self,
                                  h: HoneycombHex,
                                  *,
                                  comparison_rule_product: Tuple[Prev, ...],
                                  out_circuit: stim.Circuit,
                                  include_data_qubits: bool,
                                  tracker: MeasurementTracker):
        tracker.append_detector(
            *[
                Prev(m, offset=c.offset)
                for c in comparison_rule_product
                for m in set(h.measurement_qubits) & self.xyz_measurement_qubit_set(c.v)
            ],
            *(h.data_qubits * include_data_qubits),
            coords=[h.center.real, h.center.imag, 0],
            out_circuit=out_circuit,
        )

    def append_step_detectors(self, *,
                              cmp: Optional[ComparisonRule],
                              out_circuit: stim.Circuit,
                              tracker: MeasurementTracker):
        if cmp is None:
            return

        if cmp.kind == 'edge':
            assert cmp.product == (Prev(cmp.filter_basis, 0),)
            for edge in self.xyz_edges(cmp.filter_basis).elements:
                m = edge.measurement_qubit
                tracker.append_detector(
                    m,
                    coords=[m.real, m.imag, 0],
                    out_circuit=out_circuit,
                )
            return
        elif cmp.kind == 'hex':
            m2e = self.measure_to_element_dict

            used_hexes = [*self.xyz_hex(cmp.filter_basis)] + [
                h
                for h in self.boundary_hex_set
                if h.leaf_basis(m2e) == cmp.last_measure_basis and h.basis == cmp.filter_basis
            ]
            for h in used_hexes:
                self.append_hex_based_detector(
                    h,
                    comparison_rule_product=cmp.product,
                    out_circuit=out_circuit,
                    tracker=tracker,
                    include_data_qubits=False,
                )
        else:
            raise NotImplementedError(f'{cmp=!r}')

    @functools.cached_property
    def time_boundary_data_basis(self) -> Optional[str]:
        if self.tested_observable == 'V':
            return 'X'
        if self.tested_observable  == 'H':
            return 'Y'
        if self.tested_observable == 'EPR':
            return None
        raise NotImplementedError(f'{self.tested_observable=!r}')

    @functools.cached_property
    def use_vertical_obs(self) -> bool:
        if self.tested_observable == 'V':
            return True
        if self.tested_observable  == 'H':
            return False
        if self.tested_observable == 'EPR':
            return True
        raise NotImplementedError(f'{self.tested_observable=!r}')

    @functools.cached_property
    def use_horizontal_obs(self) -> bool:
        if self.tested_observable == 'V':
            return False
        if self.tested_observable  == 'H':
            return True
        if self.tested_observable == 'EPR':
            return True
        raise NotImplementedError(f'{self.tested_observable=!r}')

    @functools.cached_property
    def data_qubits(self) -> Tuple[complex, ...]:
        return tuple(sorted(self.data_qubit_set, key=complex_key))

    @functools.cached_property
    def measurement_qubits(self) -> Tuple[complex, ...]:
        return tuple(sorted(self.measurement_qubit_set, key=complex_key))

    def to_stabilizer_plan(self, layout_index: Optional[int]) -> StabilizerPlan:
        return StabilizerPlan(
            elements=tuple([*self.hex_plan.elements, *self.edge_plan.elements]),
            observables=() if layout_index is None else self.observable_plan(layout_index).elements,
        )

    def noisy_circuit(self) -> stim.Circuit:
        from hcb.codes.honeycomb.circuit_maker import HoneycombCircuitMaker
        maker = HoneycombCircuitMaker(layout=self)
        maker.process()
        return maker.final_noisy_circuit()

    def num_data_qubits(self) -> int:
        result = self.data_width * self.data_height
        if result % 2 == 1:
            result -= 1
        return result

    def num_used_qubits(self) -> int:
        if self.noisy_gate_set in ['EM3_v1', 'EM3_v2']:
            return self.num_data_qubits()
        if self.noisy_gate_set in ["SD6", 'SI1000']:
            nd = self.num_data_qubits()
            nm = nd // 2 * 3
            nm += self.data_width
            nm += self.data_height
            nm -= self.data_height // 3
            if self.data_width % 2 == 1 and self.data_height % 6 == 0:
                nm += 1
            return nm + nd
        raise NotImplementedError(self.noisy_gate_set)

    def to_decoding_desc(self, *, decoder: str) -> DecodingProblemDesc:
        style = 'bounded_honeycomb_memory'
        if self.sheared:
            style += '_sheared'
        style += f'_{self.noisy_gate_set}'
        return DecodingProblemDesc(
            data_width=self.data_width,
            data_height=self.data_height,
            code_distance=self.graphlike_code_distance(),
            num_qubits=self.num_used_qubits(),
            rounds=self.rounds,
            noise=self.noise_level,
            circuit_style=style,
            preserved_observable=self.tested_observable,
            decoder=decoder,
        )

    @functools.cached_property
    def ideal_and_noisy_circuit(self) -> Tuple[stim.Circuit, stim.Circuit]:
        from hcb.codes.honeycomb.circuit_maker import HoneycombCircuitMaker
        maker = HoneycombCircuitMaker(layout=self)
        maker.process()
        return (
            # Hack: get consistent result by working around not fusing during circuit concatenation.
            stim.Circuit(str(maker.final_ideal_circuit())),
            stim.Circuit(str(maker.final_noisy_circuit())),
        )

    def to_decoding_problem(self, *, decoder: str) -> DecodingProblem:
        return DecodingProblem(
            circuit_maker=lambda: self.ideal_and_noisy_circuit[1],
            desc=self.to_decoding_desc(decoder=decoder)
        )

    def to_stabilizer_plan_problem(self, *, decoder: str) -> StabilizerPlanProblem:
        ideal, noisy = self.ideal_and_noisy_circuit
        return StabilizerPlanProblem(
            ideal_circuit=ideal,
            noisy_circuit=noisy,
            all_layouts=tuple(
                self.to_stabilizer_plan(k)
                for k in range(5)
            ),
            decoding_problem=DecodingProblem(
                circuit_maker=lambda: noisy,
                desc=self.to_decoding_desc(decoder=decoder)
            ),
        )
