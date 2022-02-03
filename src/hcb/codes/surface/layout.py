import collections
import dataclasses
import re
from typing import Dict, Iterable, Optional, List, Tuple

import stim

from hcb.tools.gen.circuit_canvas import LocOp, Loc, complex_key
from hcb.tools.gen.measurement_tracker import MeasurementTracker
from hcb.tools.gen.stabilizer_plan import StabilizerPlan, StabilizerPlanElement

ul, ur, dl, dr = [e * 0.5 for e in [-1 - 1j, +1 - 1j, -1 + 1j, +1 + 1j]]
Order_ZR = [ul, ur, dl, dr]
Order_ᴎR = [ul, dl, ur, dr]
Order_NR = [dl, ul, dr, ur]
Order_SR = [dl, dr, ul, ur]
Order_ZL = Order_ZR[::-1]
Order_ᴎL = Order_ᴎR[::-1]
Order_NL = Order_NR[::-1]
Order_SL = Order_SR[::-1]
EPR_ANCILLA_LOC: complex = -3 - 0j


def checkerboard_basis(c: complex) -> str:
    """Classifies a coordinate as X type or Z type according to a checkerboard pattern."""
    return 'X' if int(c.real + c.imag) & 1 == 0 else 'Z'



def m_around(center: complex,
             basis: str,
             order: Iterable[Optional[complex]],
             *,
             outward: Optional[complex] = None) -> StabilizerPlanElement:
    """Creates a StabilizerPlanElement that measures a stabilizer using a measurement ancilla.

    Args:
        center: The measurement ancilla's position.
        basis: The Pauli basis of the stabilizer's observable.
        order: The qubits of the stabilizer's observable, listed in the order they should be
            interacted with by two qubit operations. This list can also contain None values, which
            don't indicate a qubit but instead indicate a pause; a moment during which no two qubit
            operations are performed as part of measuring this stabilizer.
        outward: Assumes `order` is a four body observable. If specified, removes the two qubits
            that are in the given direction. Must be +1, -1, +1j, -1j, or None.
    """
    if outward is None:
        drop = set()
    elif outward == -1j:
        drop = {ul, ur}
    elif outward == +1:
        drop = {ur, dr}
    elif outward == -1:
        drop = {ul, dl}
    elif outward == +1j:
        drop = {dl, dr}
    else:
        raise NotImplementedError()
    drop.add(None)
    data_order = tuple(None if offset in drop else center + offset for offset in order)
    if len(basis) == 1:
        basis = [basis] * len(data_order)
    else:
        basis = list(basis)
        assert len(basis) == len(data_order)
    return StabilizerPlanElement(
        bases=tuple(basis),
        measurement_qubit=center,
        data_qubit_order=data_order)


def rect_surface_code_plan(*,
                           width: int,
                           height: int,
                           top_left_data_qubit: complex = 0,
                           order_x: Optional[List[Optional[complex]]] = None,
                           order_z: Optional[List[Optional[complex]]] = None,
                           boundary_types: str = "XZXZ",
                           ) -> StabilizerPlan:
    """Creates a stabilizer plan corresponding to a rectangular rotated surface code.

    Args:
        width: The horizontal code distance. The number of data qubits from left to right.
        height: The vertical code distance. The number of data qubits from top to bottom.
        top_left_data_qubit: The data qubit with minimum real and imaginary coordinate.
        order_x: Data qubit touching order of X stabilizer measurements.
        order_z: Data qubit touching order of Z stabilizer measurements.
        boundary_types: Top then right then bottom then left boundary types.
    """
    top_left = top_left_data_qubit + ul
    elements: List[StabilizerPlanElement] = []

    if order_x is None:
        order_x = Order_ZR
    if order_z is None:
        order_z = Order_ᴎR
    top_boundary, right_boundary, bottom_boundary, left_boundary = boundary_types
    order_dict = {'X': order_x, 'Z': order_z}

    # Bulk stabilizers.
    for x in range(1, width):
        for y in range(1, height):
            c = x + 1j*y + top_left
            basis = checkerboard_basis(c)
            elements.append(m_around(c, basis=basis, order=order_dict[basis]))

    # Top boundary.
    for i in range(1, width):
        c = i + top_left
        if checkerboard_basis(c) == top_boundary:
            elements.append(m_around(c, basis=top_boundary, order=order_dict[top_boundary], outward=-1j))

    # Bottom boundary.
    for i in range(1, width):
        c = i + height * 1j + top_left
        if checkerboard_basis(c) == bottom_boundary:
            elements.append(m_around(c, basis=bottom_boundary, order=order_dict[bottom_boundary], outward=+1j))

    # Left boundary.
    for i in range(1, height):
        c = i*1j + top_left
        if checkerboard_basis(c) == left_boundary:
            elements.append(m_around(c, basis=left_boundary, order=order_dict[left_boundary], outward=-1))

    # Right boundary.
    for i in range(1, height):
        c = i*1j + width + top_left
        if checkerboard_basis(c) == right_boundary:
            elements.append(m_around(c, basis=right_boundary, order=order_dict[right_boundary], outward=+1))

    data_usage_counter = collections.Counter([
        q for e in elements for q in e.data_coords_set()
    ])
    clipped_data_qubits = {q for q, c in data_usage_counter.items() if c == 1}

    return StabilizerPlan(tuple(
        e.with_clipped_data_qubits(clipped_data_qubits)
        for e in elements
    ))


@dataclasses.dataclass(frozen=True)
class SingleObservableLocation:
    xs: Tuple[complex, ...]
    ys: Tuple[complex, ...]
    zs: Tuple[complex, ...]

    def all_qs(self) -> Tuple[complex, ...]:
        return self.xs + self.ys + self.zs

    def mpp_targets(self,
                    *,
                    q2i: Dict[complex, int],
                    extra_xs: Tuple[complex, ...] = (),
                    extra_ys: Tuple[complex, ...] = (),
                    extra_zs: Tuple[complex, ...] = (),
                    ) -> List[stim.GateTarget]:
        result = []
        for e in self.xs + extra_xs:
            result.append(stim.target_x(q2i[e]))
            result.append(stim.target_combiner())
        for e in self.ys + extra_ys:
            result.append(stim.target_y(q2i[e]))
            result.append(stim.target_combiner())
        for e in self.zs + extra_zs:
            result.append(stim.target_z(q2i[e]))
            result.append(stim.target_combiner())
        result.pop()
        return result

    def pauli_string(self, *, q2i: Dict[complex, int]) -> stim.PauliString:
        result = stim.PauliString(len(q2i))
        for e in self.xs:
            result[q2i[e]] = 'X'
        for e in self.ys:
            result[q2i[e]] = 'Y'
        for e in self.zs:
            result[q2i[e]] = 'Z'
        return result



@dataclasses.dataclass(frozen=True)
class AllObservableLocations:
    logical_x: SingleObservableLocation
    logical_y: SingleObservableLocation
    logical_z: SingleObservableLocation

    def logical(self, basis: str):
        if basis == "X":
            return self.logical_x
        if basis == "Y":
            return self.logical_y
        if basis == "Z":
            return self.logical_z
        raise NotImplementedError(f"{basis=!r}")


def rect_surface_code_observables(layout: StabilizerPlan) -> AllObservableLocations:
    min_r = min(q.real for q in layout.data_coords_set())
    min_i = min(q.imag for q in layout.data_coords_set())
    set_x = {q for q in layout.data_coords_set() if q.real == min_r}
    set_z = {q for q in layout.data_coords_set() if q.imag == min_i}
    obs_yx = tuple(sorted(set_x - set_z, key=complex_key))
    obs_yy = tuple(sorted(set_x & set_z, key=complex_key))
    obs_yz = tuple(sorted(set_z - set_x, key=complex_key))
    obs_x = tuple(sorted(set_x, key=complex_key))
    obs_z = tuple(sorted(set_z, key=complex_key))
    return AllObservableLocations(
        logical_x=SingleObservableLocation(xs=obs_x, ys=(), zs=()),
        logical_y=SingleObservableLocation(xs=obs_yx, ys=obs_yy, zs=obs_yz),
        logical_z=SingleObservableLocation(xs=(), ys=(), zs=obs_z),
    )


def rect_surface_code_init_circuit(*,
                                   layout: StabilizerPlan,
                                   basis: str,
                                   tracker: MeasurementTracker,
                                   q2i: Dict[complex, int],
                                   ) -> stim.Circuit:
    """Creates a logical initization circuit for the given stabilizer plan.

    The initialization circuit includes the first round of stabilizer measurements.

    Args:
        layout: A stabilizer plan returned by `rect_surface_code_plan`.
        basis: What type of initialization to perform. Allowed values are:
            "transversal_X": fault-tolerant transversal initialization of logical |+_L>.
            "transversal_Z": fault-tolerant transversal initialization of logical |0_L>.
            "frayed_Y": non-fault-tolerant initialization of |i_L>.
            /frayed_EPR_([XYZ])X_([XYZ])Z/: non-fault-tolerant initialization of a logical
                observable times X_ANC and also a second logical observable times Z_ANC.
        tracker: Measurements performed during the initialization will be recorded into this
            tracker.
        q2i: Converts qubit positions into qubit indices for stim.
    """
    canvas = layout.to_canvas()
    dq = layout.sorted_data_coords()
    circuit = stim.Circuit()
    all_obs = rect_surface_code_observables(layout)

    if basis == "transversal_X" or basis == "transversal_Z":
        # Transversal initialization.
        canvas.insert(LocOp(gate="R", targets=tuple(Loc(t=0, p=q) for q in dq)))
        if basis == "transversal_X":
            canvas.insert(LocOp(gate="H", targets=tuple(Loc(t=1, p=q) for q in dq)))
        for e in layout.sorted_elements():
            tracker.add_dummies(e.measurement_qubit, obstacle=e.common_basis() != basis[-1])
        layout.interpret_into_stim_circuit(out_tracker=tracker, q2i=q2i, edited_canvas=canvas, out_circuit=circuit)

    elif match := re.match("frayed_EPR_([XYZ])X_([XYZ])Z", basis):
        obs0 = all_obs.logical(match.group(1))
        obs1 = all_obs.logical(match.group(2))

        # Need-noiseless initialization by direct measurement of stabilizers and observables.
        circuit.append("MPP", obs0.mpp_targets(q2i=q2i, extra_xs=(EPR_ANCILLA_LOC,)))
        circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)
        circuit.append("MPP", obs1.mpp_targets(q2i=q2i, extra_zs=(EPR_ANCILLA_LOC,)))
        circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 1)
        layout.do_mpp_measurements(out_circuit=circuit, q2i=q2i, tracker=tracker)

    elif basis == "frayed_Y":
        # Need-noiseless initialization by direct measurement of stabilizers and observables.
        circuit.append("MPP", all_obs.logical_y.mpp_targets(q2i=q2i))
        circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)
        layout.do_mpp_measurements(out_circuit=circuit, q2i=q2i, tracker=tracker)

    else:
        raise NotImplementedError(f"{basis=!r}")

    return circuit


def rect_surface_code_measure_circuit(*,
                                      layout: StabilizerPlan,
                                      tracker: MeasurementTracker,
                                      q2i: Dict[complex, int],
                                      basis: str,
                                      include_obs: bool = True,
                                      ) -> stim.Circuit:
    """Creates a logical measurement circuit for the given stabilizer plan.

    The measurement circuit includes the last round of stabilizer measurements.

    Args:
        layout: A stabilizer plan returned by `rect_surface_code_plan`.
        basis: What type of measurement to perform. Allowed values are:
            "transversal_X": fault-tolerant transversal measurement of X_L.
            "transversal_Z": fault-tolerant transversal measurement of Z_L.
            "frayed_Y": non-fault-tolerant measurement of Y_L.
            /frayed_EPR_([XYZ])X_([XYZ])Z/: non-fault-tolerant measurement of a logical observable
                times X_ANC and also a second logical observable times Z_ANC.
        tracker: Measurements performed during the initialization will be recorded into this
            tracker.
        q2i: Converts qubit positions into qubit indices for stim.
        include_obs: Set to False to not add OBSERVABLE_INCLUDE instructions.
    """

    circuit = stim.Circuit()
    canvas = layout.to_canvas()
    max_t = canvas.max_time()
    dq = layout.sorted_data_coords()
    all_obs = rect_surface_code_observables(layout)

    if basis == "transversal_X" or basis == "transversal_Z":
        # Transversal measurement.
        if basis == "transversal_X":
            canvas.insert(LocOp(gate="H", targets=tuple(Loc(t=max_t - 1, p=q) for q in dq)))
        canvas.insert(LocOp(gate="M", targets=tuple(Loc(t=max_t - 0, p=q) for q in dq)))
        layout.interpret_into_stim_circuit(out_circuit=circuit, out_tracker=tracker, q2i=q2i, edited_canvas=canvas)
        for e in layout.sorted_elements():
            if e.common_basis() == basis[-1]:
                tracker.append_detector(*e.used_coords_set(),
                                        out_circuit=circuit,
                                        coords=[e.measurement_qubit.real, e.measurement_qubit.imag, 0])
        obs = all_obs.logical_x if basis == "transversal_X" else all_obs.logical_z
        if include_obs:
            circuit.append(
                "OBSERVABLE_INCLUDE",
                tracker.get_record_targets(*obs.all_qs()),
                0)

    elif match := re.match("frayed_EPR_([XYZ])X_([XYZ])Z", basis):
        obs0 = all_obs.logical(match.group(1))
        obs1 = all_obs.logical(match.group(2))

        # Need-noiseless measurement by direct measurement of stabilizers and observables.
        layout.do_mpp_measurements(
            out_circuit=circuit,
            q2i=q2i,
            tracker=tracker,
            detect_vs_prev=True)

        if include_obs:
            circuit.append("MPP", obs0.mpp_targets(q2i=q2i, extra_xs=(EPR_ANCILLA_LOC,)))
            circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)
            circuit.append("MPP", obs1.mpp_targets(q2i=q2i, extra_zs=(EPR_ANCILLA_LOC,)))
            circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 1)
    elif basis == "frayed_Y":
        # Need-noiseless measurement by direct measurement of stabilizers and observables.
        layout.do_mpp_measurements(
            out_circuit=circuit,
            q2i=q2i,
            tracker=tracker,
            detect_vs_prev=True)

        if include_obs:
            circuit.append("MPP", all_obs.logical_y.mpp_targets(q2i=q2i))
            circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)
    else:
        raise NotImplementedError(f"{basis=!r}")

    return circuit


def interpret_layer_removing_data_qubits(
        *,
        q2i: Dict[complex, int],
        layout: StabilizerPlan,
        tracker: MeasurementTracker,
        removed_data_qubits: Iterable[complex],
        out_circuit: stim.Circuit,
        removed_basis: str) -> None:
    assert removed_basis in "XZ"
    removed_data_sorted = sorted(removed_data_qubits, key=complex_key)
    removed_data_set = set(removed_data_sorted)
    canvas = layout.to_canvas()
    max_t = canvas.max_time()
    if removed_basis == "X":
        canvas.insert(LocOp(gate="H", targets=tuple(Loc(t=max_t - 1, p=q) for q in removed_data_sorted)))
    canvas.insert(LocOp(gate="M", targets=tuple(Loc(t=max_t - 0, p=q) for q in removed_data_sorted)))
    layout.interpret_into_stim_circuit(out_circuit=out_circuit, out_tracker=tracker, q2i=q2i, edited_canvas=canvas)
    for q in layout.elements:
        if q.common_basis() == removed_basis:
            lost_qubits = q.data_coords_set() & removed_data_set
            if lost_qubits:
                if lost_qubits == q.data_coords_set():
                    tracker.append_detector(*lost_qubits, q.measurement_qubit, out_circuit=out_circuit,
                                            coords=[q.measurement_qubit.real, q.measurement_qubit.imag, 0])
                else:
                    tracker.add_group(*lost_qubits, q.measurement_qubit, group_key=q.measurement_qubit)
