import stim

from hcb.codes.surface.layout import rect_surface_code_plan, AllObservableLocations, SingleObservableLocation, \
    rect_surface_code_observables, rect_surface_code_init_circuit, EPR_ANCILLA_LOC
from hcb.tools.gen.circuit_canvas import indexed_qubits_circuit_dict
from hcb.tools.gen.measurement_tracker import MeasurementTracker


def test_rect_surface_code_observables():
    plan = rect_surface_code_plan(width=3, height=5)
    obs = rect_surface_code_observables(plan)
    assert obs == AllObservableLocations(
        logical_x=SingleObservableLocation(
            xs=(0, 1j, 2j, 3j, 4j),
            ys=(),
            zs=(),
        ),
        logical_y=SingleObservableLocation(
            xs=(1j, 2j, 3j, 4j),
            ys=(0,),
            zs=(1, 2),
        ),
        logical_z=SingleObservableLocation(
            xs=(),
            ys=(),
            zs=(0, 1, 2),
        ),
    )


def test_rect_surface_code_init_circuit_transversal_X():
    plan = rect_surface_code_plan(width=3, height=3)
    _, q2i = indexed_qubits_circuit_dict(plan.used_coords_set())
    tracker = MeasurementTracker()
    circuit = rect_surface_code_init_circuit(layout=plan,
                                             basis="transversal_X",
                                             tracker=tracker,
                                             q2i=q2i)
    assert len(tracker.history) == len(plan.measure_coords_set())
    assert circuit == stim.Circuit("""
        R 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
        TICK
        H 0 1 2 3 4 5 6 7 8 10 12 13 15
        TICK
        CX 0 11 4 14 6 16 12 1 13 3 15 5
        TICK
        CX 1 11 5 14 7 16 12 4 13 6 15 8
        TICK
        CX 1 9 3 11 7 14 10 0 12 2 13 4
        TICK
        CX 2 9 4 11 8 14 10 3 12 5 13 7
        TICK
        H 10 12 13 15
        TICK
        M 9 10 11 12 13 14 15 16
        DETECTOR(0.5, -0.5, 0) rec[-7]
        DETECTOR(0.5, 1.5, 0) rec[-5]
        DETECTOR(1.5, 0.5, 0) rec[-4]
        DETECTOR(1.5, 2.5, 0) rec[-2]
        SHIFT_COORDS(0, 0, 1)
    """)


def test_rect_surface_code_init_circuit_transversal_Z():
    plan = rect_surface_code_plan(width=3, height=3)
    _, q2i = indexed_qubits_circuit_dict(plan.used_coords_set())
    tracker = MeasurementTracker()
    circuit = rect_surface_code_init_circuit(layout=plan,
                                             basis="transversal_Z",
                                             tracker=tracker,
                                             q2i=q2i)
    assert len(tracker.history) == len(plan.measure_coords_set())
    assert circuit == stim.Circuit("""
        R 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
        TICK
        H 10 12 13 15
        TICK
        CX 0 11 4 14 6 16 12 1 13 3 15 5
        TICK
        CX 1 11 5 14 7 16 12 4 13 6 15 8
        TICK
        CX 1 9 3 11 7 14 10 0 12 2 13 4
        TICK
        CX 2 9 4 11 8 14 10 3 12 5 13 7
        TICK
        H 10 12 13 15
        TICK
        M 9 10 11 12 13 14 15 16
        DETECTOR(-0.5, 1.5, 0) rec[-8]
        DETECTOR(0.5, 0.5, 0) rec[-6]
        DETECTOR(1.5, 1.5, 0) rec[-3]
        DETECTOR(2.5, 0.5, 0) rec[-1]
        SHIFT_COORDS(0, 0, 1)
    """)


def test_rect_surface_code_init_circuit_frayed_Y():
    plan = rect_surface_code_plan(width=3, height=3)
    _, q2i = indexed_qubits_circuit_dict(plan.used_coords_set())
    tracker = MeasurementTracker()
    circuit = rect_surface_code_init_circuit(layout=plan,
                                             basis="frayed_Y",
                                             tracker=tracker,
                                             q2i=q2i)
    assert len(tracker.history) == len(plan.measure_coords_set())
    assert circuit == stim.Circuit("""
        MPP X1*X2*Y0*Z3*Z6
        OBSERVABLE_INCLUDE(0) rec[-1]
        MPP Z1*Z2 X0*X3 Z0*Z1*Z3*Z4 X1*X4*X2*X5 X3*X6*X4*X7 Z4*Z5*Z7*Z8 X5*X8 Z6*Z7
    """)


def test_rect_surface_code_init_circuit_frayed_EPR_XX_ZZ():
    plan = rect_surface_code_plan(width=3, height=3)
    _, q2i = indexed_qubits_circuit_dict(plan.used_coords_set() | {EPR_ANCILLA_LOC})
    tracker = MeasurementTracker()
    circuit = rect_surface_code_init_circuit(layout=plan,
                                             basis="frayed_EPR_XX_ZZ",
                                             tracker=tracker,
                                             q2i=q2i)
    assert len(tracker.history) == len(plan.measure_coords_set())
    assert circuit == stim.Circuit("""
        MPP X1*X2*X3*X0
        OBSERVABLE_INCLUDE(0) rec[-1]
        MPP Z1*Z4*Z7*Z0
        OBSERVABLE_INCLUDE(1) rec[-1]
        MPP Z2*Z3 X1*X4 Z1*Z2*Z4*Z5 X2*X5*X3*X6 X4*X7*X5*X8 Z5*Z6*Z8*Z9 X6*X9 Z7*Z8
    """)


def test_rect_surface_code_init_circuit_frayed_EPR_YX_ZZ():
    plan = rect_surface_code_plan(width=3, height=3)
    _, q2i = indexed_qubits_circuit_dict(plan.used_coords_set() | {EPR_ANCILLA_LOC})
    tracker = MeasurementTracker()
    circuit = rect_surface_code_init_circuit(layout=plan,
                                             basis="frayed_EPR_YX_ZZ",
                                             tracker=tracker,
                                             q2i=q2i)
    assert len(tracker.history) == len(plan.measure_coords_set())
    assert circuit == stim.Circuit("""
        MPP X2*X3*X0*Y1*Z4*Z7
        OBSERVABLE_INCLUDE(0) rec[-1]
        MPP Z1*Z4*Z7*Z0
        OBSERVABLE_INCLUDE(1) rec[-1]
        MPP Z2*Z3 X1*X4 Z1*Z2*Z4*Z5 X2*X5*X3*X6 X4*X7*X5*X8 Z5*Z6*Z8*Z9 X6*X9 Z7*Z8
    """)
