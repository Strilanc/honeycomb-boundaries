import pathlib
from typing import Dict, List, Any

import stim

from hcb.codes.honeycomb.layout import (
    HoneycombLayout,
    comparisons_for_step,
    EDGE_MEASUREMENT_SEQUENCE, checkerboard_type, SI1000_DATA_ROTATION_SEQUENCE,
)
from hcb.tools.gen.circuit_canvas import complex_key
from hcb.tools.gen.measurement_tracker import MeasurementTracker, Prev
from hcb.tools.gen.noise import NoiseModel
from hcb.tools.gen.stabilizer_plan import StabilizerPlan, StabilizerPlanElement
from hcb.tools.gen.viewer import stim_circuit_html_viewer


EPR_ANCILLA: complex = -2


def rotation_op(old_basis: str, next_basis: str) -> str:
    if old_basis == next_basis:
        return 'I'
    suffix = ''.join(sorted([old_basis, next_basis]))
    return f'H_{suffix}'


def complex_key_prefer_ints(c: complex) -> Any:
    return int(c.real) != c.real or int(c.imag) != c.imag, complex_key(c)


class HoneycombCircuitMaker:
    def __init__(self, *, layout: HoneycombLayout):
        self.noiseless_head_moments: List[stim.Circuit] = []
        self.moments: List[stim.Circuit] = []
        self.noiseless_tail_moments: List[stim.Circuit] = []
        self.tracker = MeasurementTracker()
        self.q2i: Dict[complex, int] = {}
        self.measured_bases = ''
        self.layout = layout

    def process(self):
        if self.layout.noisy_gate_set.startswith('EM3'):
            used_qubits = self.layout.edge_plan.data_coords_set()
        elif self.layout.noisy_gate_set in ['SD6', 'SI1000']:
            used_qubits = self.layout.edge_plan.used_coords_set()
        else:
            raise NotImplementedError()
        self.q2i = {q: i for i, q in enumerate(sorted(used_qubits, key=complex_key_prefer_ints))}
        if self.layout.tested_observable == 'EPR':
            assert EPR_ANCILLA not in self.q2i
            self.q2i[EPR_ANCILLA] = len(self.q2i)

        # Initialization round pair.
        self.append_init_round_pair()

        # Repeated round pair.
        loop_moments = []
        self.append_round_pair(out_moments=loop_moments)
        loop = stim.Circuit()
        for loop_moment in loop_moments:
            loop += loop_moment
            loop.append("TICK")
        self.moments.append(loop * ((self.layout.rounds - 4) // 2))

        # Measurement round pair.
        self.append_round_pair_measure()

    def coords_header(self) -> stim.Circuit:
        result = stim.Circuit()
        for q, i in self.q2i.items():
            result.append("QUBIT_COORDS", i, [q.real, q.imag])
        return result

    def is_physical(self) -> bool:
        return not self.noiseless_head_moments and not self.noiseless_tail_moments

    def final_ideal_circuit(self) -> stim.Circuit:
        result = self.coords_header()
        for c in self.noiseless_head_moments:
            result += c
            result.append("TICK")
        for c in self.moments:
            result += c
            if len(result) != 1 or not isinstance(result[0], stim.CircuitRepeatBlock):
                result.append("TICK")
        for c in self.noiseless_tail_moments:
            result += c
            result.append("TICK")
        return result

    def final_noisy_circuit(self) -> stim.Circuit:
        result = self.coords_header()
        for c in self.noiseless_head_moments:
            result += c
            result.append("TICK")

        noisy_part = stim.Circuit()
        for c in self.moments:
            noisy_part += c
            if len(result) != 1 or not isinstance(result[0], stim.CircuitRepeatBlock):
                noisy_part.append("TICK")
        result += self.noise_model().noisy_circuit(noisy_part)

        for c in self.noiseless_tail_moments:
            result += c
            result.append("TICK")
        return result

    def noise_model(self) -> NoiseModel:
        if self.layout.noisy_gate_set == 'EM3_v2':
            return NoiseModel.EM3_v2(self.layout.noise_level)
        if self.layout.noisy_gate_set == 'EM3_v1':
            return NoiseModel.EM3_v1(self.layout.noise_level)
        if self.layout.noisy_gate_set == 'SD6':
            return NoiseModel.SD6(self.layout.noise_level)
        if self.layout.noisy_gate_set == 'SI1000':
            return NoiseModel.SI1000(self.layout.noise_level)
        raise NotImplementedError()

    def append_init_round_pair(self):
        if self.layout.tested_observable == 'EPR':
            self.noiseless_head_moments.append(magical_obs_bell_measurement(
                obs=self.layout.vertical_observable(layout_index=1),
                obs_index=0,
                q2i=self.q2i))
            self.noiseless_head_moments.append(magical_obs_bell_measurement(
                obs=self.layout.horizontal_observable(layout_index=1),
                obs_index=1,
                q2i=self.q2i))
            self.append_round_pair(out_moments=self.noiseless_head_moments)
            return

        assert self.layout.tested_observable in ['H', 'V']
        assert not self.moments
        data_targets = [self.q2i[q] for q in self.layout.data_qubits]
        self.moments.append(stim.Circuit())
        self.moments[0].append("R", data_targets)
        self.append_round_pair(out_moments=self.moments)
        self.moments.insert(1, stim.Circuit())
        self.moments[1].append(f"H_{self.layout.time_boundary_data_basis}Z", data_targets)

    def append_round_pair(self, *, out_moments: List[stim.Circuit]):
        if self.layout.noisy_gate_set.startswith('EM3'):
            self.append_em_round_pair(out_moments=out_moments)
        elif self.layout.noisy_gate_set == 'SD6':
            self.append_sd6_round_pair(out_moments=out_moments)
        elif self.layout.noisy_gate_set == 'SI1000':
            self.append_si1000_round_pair(out_moments=out_moments)
        else:
            raise NotImplementedError(f'{self.layout.noisy_gate_set}')

    def append_em_round_pair(self, *, out_moments: List[stim.Circuit]):
        for edge_basis in EDGE_MEASUREMENT_SEQUENCE:
            out_moments.append(stim.Circuit())
            for edge in self.layout.xyz_edges(edge_basis).elements:
                edge.append_mpp(out_circuit=out_moments[-1], q2i=self.q2i)
            self.finish_edge_measurements(
                edge_basis=edge_basis,
                out_moments=out_moments,
                add_operations=False)

    def append_sd6_round_pair(self, *, out_moments: List[stim.Circuit]):
        data_targets = [self.q2i[q] for q in self.layout.data_qubits]
        for edge_basis in EDGE_MEASUREMENT_SEQUENCE:
            edges = self.layout.xyz_edges(edge_basis).elements
            measure_targets = [self.q2i[e.measurement_qubit] for e in edges]

            out_moments.append(stim.Circuit())
            out_moments[-1].append('R', measure_targets)
            if edge_basis != 'Z':
                out_moments[-1].append(f'H_{edge_basis}Z', data_targets)

            out_moments.append(stim.Circuit())
            out_moments[-1].append('CX', [
                self.q2i[q]
                for e in edges
                if e.data_qubit_order[0] is not None
                for q in [e.data_qubit_order[0], e.measurement_qubit]
            ])

            out_moments.append(stim.Circuit())
            out_moments[-1].append('CX', [
                self.q2i[q]
                for e in edges
                if e.data_qubit_order[1] is not None
                for q in [e.data_qubit_order[1], e.measurement_qubit]
            ])

            out_moments.append(stim.Circuit())
            if edge_basis != 'Z':
                out_moments[-1].append(f'H_{edge_basis}Z', data_targets)
            self.finish_edge_measurements(edge_basis=edge_basis,
                                          out_moments=out_moments,
                                          add_operations=True)

    def append_si1000_round_pair(self, *, out_moments: List[stim.Circuit]):
        data_targets = [self.q2i[q] for q in self.layout.data_qubits]
        data_targets_0 = [self.q2i[q] for q in self.layout.data_qubits if not checkerboard_type(q)]
        data_targets_1 = [self.q2i[q] for q in self.layout.data_qubits if checkerboard_type(q)]
        parts = [EDGE_MEASUREMENT_SEQUENCE[:3], EDGE_MEASUREMENT_SEQUENCE[3:]]
        all_measure_targets = [self.q2i[q] for q in self.layout.measurement_qubits]

        def cz_layer(*, basis: str, parity: int) -> stim.Circuit:
            layer = stim.Circuit()
            layer.append('CZ', [
                self.q2i[q]
                for e in self.layout.xyz_edges(basis).elements
                if e.data_qubit_order[parity] is not None
                for q in [e.data_qubit_order[parity], e.measurement_qubit]
            ])
            return layer

        for (a, b, c), (r_a, rab, rbc, rc_) in zip(parts, SI1000_DATA_ROTATION_SEQUENCE):
            out_moments.append(stim.Circuit())
            out_moments[-1].append('R', all_measure_targets)

            out_moments.append(stim.Circuit())
            out_moments[-1].append('H', all_measure_targets)
            if r_a != 'I':
                out_moments[-1].append(r_a, data_targets)

            out_moments.append(cz_layer(basis=a, parity=0))

            out_moments.append(cz_layer(basis=a, parity=1))
            out_moments[-1].append(rab, data_targets_0)

            out_moments.append(cz_layer(basis=b, parity=0))
            out_moments[-1].append(rab, data_targets_1)

            out_moments.append(cz_layer(basis=b, parity=1))
            out_moments[-1].append(rbc, data_targets_0)

            out_moments.append(cz_layer(basis=c, parity=0))
            out_moments[-1].append(rbc, data_targets_1)

            out_moments.append(cz_layer(basis=c, parity=1))

            out_moments.append(stim.Circuit())
            out_moments[-1].append('H', all_measure_targets)
            if rc_ != 'I':
                out_moments[-1].append(rc_, data_targets)

            out_moments.append(stim.Circuit())
            for edge_basis in a, b, c:
                self.finish_edge_measurements(edge_basis=edge_basis,
                                              out_moments=out_moments,
                                              add_operations=True)

    def finish_edge_measurements(self,
                                 *,
                                 edge_basis: str,
                                 out_moments: List[stim.Circuit],
                                 add_operations: bool):
        added_measurements = [edge.measurement_qubit for edge in self.layout.xyz_edges(edge_basis).elements]
        if add_operations:
            out_moments[-1].append('M', [self.q2i[q] for q in added_measurements])
        self.tracker.add_measurements(*added_measurements)

        # Move logical observables.
        if edge_basis != 'X':  # Because X basis is the turn-around basis.
            if self.layout.use_vertical_obs:
                out_moments[-1].append(
                    "OBSERVABLE_INCLUDE",
                    self.tracker.get_record_targets(
                        *(set(added_measurements) & self.layout.vertical_observable_measurement_qubit_set)),
                    0)
            if self.layout.use_horizontal_obs:
                out_moments[-1].append(
                    "OBSERVABLE_INCLUDE",
                    self.tracker.get_record_targets(
                        *(set(added_measurements) & self.layout.horizontal_observable_measurement_qubit_set)),
                    1)

        # Compare to previous measurements when possible.
        self.measured_bases += edge_basis
        self.layout.append_step_detectors(
            cmp=comparisons_for_step(
                previous_measurements=self.measured_bases,
                data_init_basis=self.layout.time_boundary_data_basis,
                data_measure_basis=None,
            ),
            out_circuit=out_moments[-1],
            tracker=self.tracker,
        )
        out_moments[-1].append("SHIFT_COORDS", [], [0, 0, 1])

    def append_round_pair_measure(self):
        layout = self.layout
        if layout.tested_observable == 'EPR':
            self.append_round_pair(out_moments=self.noiseless_tail_moments)
            self.noiseless_tail_moments.append(magical_obs_bell_measurement(
                obs=layout.vertical_observable(layout_index=1),
                obs_index=0,
                q2i=self.q2i))
            self.noiseless_tail_moments.append(magical_obs_bell_measurement(
                obs=layout.horizontal_observable(layout_index=1),
                obs_index=1,
                q2i=self.q2i))
            return

        assert layout.tested_observable in ['H', 'V']
        self.append_round_pair(out_moments=self.moments)
        data_targets = [self.q2i[q] for q in layout.data_qubits]
        self.moments.append(stim.Circuit())
        self.moments.append(stim.Circuit())
        self.moments[-2].append(f"H_{layout.time_boundary_data_basis}Z", data_targets)
        self.moments[-1].append("M", data_targets)
        self.tracker.add_measurements(*layout.data_qubits)

        # Hardcoded detectors at the future time boundary.
        if layout.tested_observable == 'V':
            obs_getter = layout.vertical_observable
            obs_index = 0
            m2e = layout.measure_to_element_dict
            for h in [*layout.hexes, *layout.boundary_hex_set]:
                if h.basis == 'X' and h.leaf_basis(m2e=m2e) in [None, 'Z']:
                    self.tracker.append_detector(
                        *h.measurement_qubits,
                        *h.data_qubits,
                        coords=[h.center.real, h.center.imag, 1],
                        out_circuit=self.moments[-1],
                    )
            for h in layout.xyz_hex('Z'):
                layout.append_hex_based_detector(
                    h,
                    comparison_rule_product=(
                        Prev('Y', 0),
                        Prev('Y', 1),
                        Prev('X', 1),
                    ),
                    include_data_qubits=True,
                    tracker=self.tracker,
                    out_circuit=self.moments[-1],
                )
        else:
            assert layout.tested_observable == 'H'
            obs_getter = layout.horizontal_observable
            obs_index = 1
            for edge in layout.xyz_edges('Y').elements:
                m = edge.measurement_qubit
                self.tracker.append_detector(
                    m,
                    *edge.data_coords_set(),
                    coords=[m.real, m.imag, 0],
                    out_circuit=self.moments[-1],
                )
            for h in layout.xyz_hex('Y'):
                self.tracker.append_detector(
                    *h.measurement_qubits,
                    *h.data_qubits,
                    coords=[h.center.real, h.center.imag, 0],
                    out_circuit=self.moments[-1],
                )
        self.moments[-1].append(
            "OBSERVABLE_INCLUDE",
            self.tracker.get_record_targets(*obs_getter(layout_index=1).xyz_data_coords()),
            obs_index)


def magical_obs_bell_measurement(*,
                                 obs: StabilizerPlanElement,
                                 obs_index: int,
                                 q2i: Dict[complex, int],
                                 ) -> stim.Circuit:
    target_xyz = {
        'X': stim.target_x,
        'Y': stim.target_y,
        'Z': stim.target_z,
    }
    mpp_targets = []
    for p, q in zip(obs.bases, obs.data_qubit_order):
        assert p in '_XYZ'
        if p != '_':
            mpp_targets.append(target_xyz[p](q2i[q]))
            mpp_targets.append(stim.target_combiner())
    mpp_targets.append(target_xyz['XZ'[obs_index]](q2i[EPR_ANCILLA]))
    circuit = stim.Circuit()
    circuit.append('MPP', mpp_targets)
    circuit.append('OBSERVABLE_INCLUDE', stim.target_rec(-1), obs_index)
    return circuit


def main():
    out_dir = pathlib.Path(__file__).parent.parent.parent.parent.parent / 'out'
    layout = HoneycombLayout(data_width=8,
                             data_height=12,
                             rounds=10,
                             noise_level=0.001,
                             noisy_gate_set='SI1000',
                             tested_observable='H',
                             sheared=False)
    edge_plan = layout.edge_plan
    hex_plan = layout.hex_plan
    plans = []
    for step in range(5):
        observable_plan = layout.observable_plan(step)
        plans.append(StabilizerPlan(
            elements=tuple([*hex_plan.elements, *edge_plan.elements]),
            observables=observable_plan.observables,
        ))

    with open(out_dir / 'tmp.svg', 'w') as f:
        print(StabilizerPlan.svg(*plans, show_order=False), file=f)

    circuit = layout.noisy_circuit()
    with open(out_dir / 'circuit.stim', 'w') as f:
        print(circuit, file=f)
    with open(out_dir / 'tmp.html', 'w') as f:
        print(stim_circuit_html_viewer(circuit=circuit, width=500, height=500), file=f)
    error_model: stim.DetectorErrorModel = circuit.detector_error_model(decompose_errors=False)

    shortest_error = error_model.shortest_graphlike_error(ignore_ungraphlike_errors=True)
    with open(out_dir / 'filter.dem', 'w') as f:
        print(shortest_error, file=f)
    for e in shortest_error:
        print("    ", e)
    print(f"graphlike code distance = {len(shortest_error)}")

    # shortest_error_circuit = circuit.shortest_graphlike_error(ignore_ungraphlike_errors=True)
    # print("Circuit equivalents")
    # for e in shortest_error_circuit:
    #     print("    " + e.replace('\n', '\n    '))
    # print(f"graphlike code distance = {len(shortest_error)}")


if __name__ == '__main__':
    main()
