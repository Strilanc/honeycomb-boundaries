import pytest
import stim

from hcb.codes.honeycomb.layout import HoneycombLayout


@pytest.mark.parametrize("data_width,data_height,rounds,gate_set,tested_observable,expected_graphlike_code_distance", [
    (8, 12, 10, 'EM3_v1', 'H', 4),
    (8, 15, 10, 'EM3_v1', 'H', 5),
    (10, 12, 10, 'EM3_v1', 'H', 4),

    (8, 12, 10, 'EM3_v1', 'V', 4),
    (8, 15, 10, 'EM3_v1', 'V', 4),
    (10, 12, 10, 'EM3_v1', 'V', 5),

    (8, 12, 10, 'EM3_v1', 'EPR', 4),
    (8, 15, 10, 'EM3_v1', 'EPR', 4),
    (10, 12, 10, 'EM3_v1', 'EPR', 4),
    (10, 15, 10, 'EM3_v1', 'EPR', 5),

    (8, 12, 10, 'EM3_v2', 'EPR', 4),
])
def test_graphlike_code_distances(data_width: int,
                                  data_height: int,
                                  rounds: int,
                                  gate_set: str,
                                  tested_observable: str,
                                  expected_graphlike_code_distance: int):
    layout = HoneycombLayout(data_width=data_width,
                             data_height=data_height,
                             rounds=rounds,
                             noise_level=0.001,
                             noisy_gate_set=gate_set,
                             tested_observable=tested_observable)
    circuit = layout.noisy_circuit()
    dem = circuit.detector_error_model(decompose_errors=True)
    err = dem.shortest_graphlike_error()
    assert len(err) == expected_graphlike_code_distance


def test_exact_circuit_EPR():
    layout = HoneycombLayout(data_width=2,
                             data_height=6,
                             rounds=100,
                             noise_level=0.125,
                             noisy_gate_set='EM3_v1',
                             tested_observable='H')
    assert layout.to_problem(decoder='pymatching').noisy_circuit == stim.Circuit("""
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        QUBIT_COORDS(1, 1) 2
        QUBIT_COORDS(1, 2) 3
        QUBIT_COORDS(1, 3) 4
        QUBIT_COORDS(2, 1) 5
        QUBIT_COORDS(2, 2) 6
        QUBIT_COORDS(2, 3) 7
        QUBIT_COORDS(2, 4) 8
        QUBIT_COORDS(2, 5) 9
        QUBIT_COORDS(3, 4) 10
        QUBIT_COORDS(3, 5) 11
        R 0 1 2 3 4 5 6 7 8 9 10 11
        X_ERROR(0.125) 0 1 2 3 4 5 6 7 8 9 10 11
        TICK
        H_YZ 0 1 2 3 4 5 6 7 8 9 10 11
        DEPOLARIZE1(0.125) 0 1 2 3 4 5 6 7 8 9 10 11
        TICK
        DEPOLARIZE2(0.125) 2 3 5 6 8 9 10 11 0 1 4 7
        MPP(0.125) X2*X3 X5*X6 X8*X9 X10*X11 X0*X1 X4*X7
        TICK
        DEPOLARIZE1(0.125) 0 4 5 10 3 9 6 11
        DEPOLARIZE2(0.125) 1 2 7 8
        MPP(0.125) Y0 Y1*Y2 Y4 Y5 Y7*Y8 Y10 Y3 Y9 Y6 Y11
        OBSERVABLE_INCLUDE(1) rec[-4] rec[-2]
        DETECTOR(1.5, 4, 0) rec[-8] rec[-6] rec[-3]
        DETECTOR(2.5, 1, 0) rec[-7] rec[-2]
        DETECTOR(0.5, 1, 0) rec[-10] rec[-9] rec[-4]
        DETECTOR(3.5, 4, 0) rec[-5] rec[-1]
        TICK
        DEPOLARIZE1(0.125) 0 1 9 11
        DEPOLARIZE2(0.125) 3 4 6 7 2 5 8 10
        MPP(0.125) Z0 Z1 Z3*Z4 Z6*Z7 Z9 Z11 Z2*Z5 Z8*Z10
        OBSERVABLE_INCLUDE(1) rec[-2]
        TICK
        DEPOLARIZE2(0.125) 2 3 5 6 8 9 10 11 0 1 4 7
        MPP(0.125) X2*X3 X5*X6 X8*X9 X10*X11 X0*X1 X4*X7
        DETECTOR(1.5, 2, 0) rec[-12] rec[-11] rec[-8] rec[-6] rec[-5] rec[-1]
        TICK
        DEPOLARIZE1(0.125) 0 1 9 11
        DEPOLARIZE2(0.125) 3 4 6 7 2 5 8 10
        MPP(0.125) Z0 Z1 Z3*Z4 Z6*Z7 Z9 Z11 Z2*Z5 Z8*Z10
        OBSERVABLE_INCLUDE(1) rec[-2]
        DETECTOR(1.5, 2, 0) rec[-20] rec[-19] rec[-16] rec[-6] rec[-5] rec[-2]
        DETECTOR(2.5, 5, 0) rec[-18] rec[-17] rec[-15] rec[-4] rec[-3] rec[-1]
        DETECTOR(0.5, -1, 0) rec[-22] rec[-21] rec[-8] rec[-7]
        TICK
        DEPOLARIZE1(0.125) 0 4 5 10 3 9 6 11
        DEPOLARIZE2(0.125) 1 2 7 8
        MPP(0.125) Y0 Y1*Y2 Y4 Y5 Y7*Y8 Y10 Y3 Y9 Y6 Y11
        OBSERVABLE_INCLUDE(1) rec[-4] rec[-2]
        DETECTOR(0.5, 3, 0) rec[-40] rec[-36] rec[-30] rec[-16] rec[-8] rec[-4]
        DETECTOR(2.5, 3, 0) rec[-38] rec[-37] rec[-34] rec[-29] rec[-25] rec[-15] rec[-11] rec[-6] rec[-5] rec[-2]
        TICK
        REPEAT 48 {
            DEPOLARIZE2(0.125) 2 3 5 6 8 9 10 11 0 1 4 7
            MPP(0.125) X2*X3 X5*X6 X8*X9 X10*X11 X0*X1 X4*X7
            TICK
            DEPOLARIZE1(0.125) 0 4 5 10 3 9 6 11
            DEPOLARIZE2(0.125) 1 2 7 8
            MPP(0.125) Y0 Y1*Y2 Y4 Y5 Y7*Y8 Y10 Y3 Y9 Y6 Y11
            OBSERVABLE_INCLUDE(1) rec[-4] rec[-2]
            DETECTOR(1.5, 4, 0) rec[-24] rec[-22] rec[-19] rec[-8] rec[-6] rec[-3]
            DETECTOR(2.5, 1, 0) rec[-23] rec[-18] rec[-7] rec[-2]
            DETECTOR(0.5, 1, 0) rec[-26] rec[-25] rec[-20] rec[-10] rec[-9] rec[-4]
            DETECTOR(3.5, 4, 0) rec[-21] rec[-17] rec[-5] rec[-1]
            TICK
            DEPOLARIZE1(0.125) 0 1 9 11
            DEPOLARIZE2(0.125) 3 4 6 7 2 5 8 10
            MPP(0.125) Z0 Z1 Z3*Z4 Z6*Z7 Z9 Z11 Z2*Z5 Z8*Z10
            OBSERVABLE_INCLUDE(1) rec[-2]
            TICK
            DEPOLARIZE2(0.125) 2 3 5 6 8 9 10 11 0 1 4 7
            MPP(0.125) X2*X3 X5*X6 X8*X9 X10*X11 X0*X1 X4*X7
            DETECTOR(1.5, 2, 0) rec[-54] rec[-53] rec[-49] rec[-46] rec[-45] rec[-42] rec[-12] rec[-11] rec[-8] rec[-6] rec[-5] rec[-1]
            TICK
            DEPOLARIZE1(0.125) 0 1 9 11
            DEPOLARIZE2(0.125) 3 4 6 7 2 5 8 10
            MPP(0.125) Z0 Z1 Z3*Z4 Z6*Z7 Z9 Z11 Z2*Z5 Z8*Z10
            OBSERVABLE_INCLUDE(1) rec[-2]
            DETECTOR(1.5, 2, 0) rec[-20] rec[-19] rec[-16] rec[-6] rec[-5] rec[-2]
            DETECTOR(2.5, 5, 0) rec[-18] rec[-17] rec[-15] rec[-4] rec[-3] rec[-1]
            DETECTOR(0.5, -1, 0) rec[-22] rec[-21] rec[-8] rec[-7]
            TICK
            DEPOLARIZE1(0.125) 0 4 5 10 3 9 6 11
            DEPOLARIZE2(0.125) 1 2 7 8
            MPP(0.125) Y0 Y1*Y2 Y4 Y5 Y7*Y8 Y10 Y3 Y9 Y6 Y11
            OBSERVABLE_INCLUDE(1) rec[-4] rec[-2]
            DETECTOR(0.5, 3, 0) rec[-40] rec[-36] rec[-30] rec[-16] rec[-8] rec[-4]
            DETECTOR(2.5, 3, 0) rec[-38] rec[-37] rec[-34] rec[-29] rec[-25] rec[-15] rec[-11] rec[-6] rec[-5] rec[-2]
            TICK
        }
        TICK
        DEPOLARIZE2(0.125) 2 3 5 6 8 9 10 11 0 1 4 7
        MPP(0.125) X2*X3 X5*X6 X8*X9 X10*X11 X0*X1 X4*X7
        TICK
        DEPOLARIZE1(0.125) 0 4 5 10 3 9 6 11
        DEPOLARIZE2(0.125) 1 2 7 8
        MPP(0.125) Y0 Y1*Y2 Y4 Y5 Y7*Y8 Y10 Y3 Y9 Y6 Y11
        OBSERVABLE_INCLUDE(1) rec[-4] rec[-2]
        DETECTOR(1.5, 4, 0) rec[-24] rec[-22] rec[-19] rec[-8] rec[-6] rec[-3]
        DETECTOR(2.5, 1, 0) rec[-23] rec[-18] rec[-7] rec[-2]
        DETECTOR(0.5, 1, 0) rec[-26] rec[-25] rec[-20] rec[-10] rec[-9] rec[-4]
        DETECTOR(3.5, 4, 0) rec[-21] rec[-17] rec[-5] rec[-1]
        TICK
        DEPOLARIZE1(0.125) 0 1 9 11
        DEPOLARIZE2(0.125) 3 4 6 7 2 5 8 10
        MPP(0.125) Z0 Z1 Z3*Z4 Z6*Z7 Z9 Z11 Z2*Z5 Z8*Z10
        OBSERVABLE_INCLUDE(1) rec[-2]
        TICK
        DEPOLARIZE2(0.125) 2 3 5 6 8 9 10 11 0 1 4 7
        MPP(0.125) X2*X3 X5*X6 X8*X9 X10*X11 X0*X1 X4*X7
        DETECTOR(1.5, 2, 0) rec[-54] rec[-53] rec[-49] rec[-46] rec[-45] rec[-42] rec[-12] rec[-11] rec[-8] rec[-6] rec[-5] rec[-1]
        TICK
        DEPOLARIZE1(0.125) 0 1 9 11
        DEPOLARIZE2(0.125) 3 4 6 7 2 5 8 10
        MPP(0.125) Z0 Z1 Z3*Z4 Z6*Z7 Z9 Z11 Z2*Z5 Z8*Z10
        OBSERVABLE_INCLUDE(1) rec[-2]
        DETECTOR(1.5, 2, 0) rec[-20] rec[-19] rec[-16] rec[-6] rec[-5] rec[-2]
        DETECTOR(2.5, 5, 0) rec[-18] rec[-17] rec[-15] rec[-4] rec[-3] rec[-1]
        DETECTOR(0.5, -1, 0) rec[-22] rec[-21] rec[-8] rec[-7]
        TICK
        DEPOLARIZE1(0.125) 0 4 5 10 3 9 6 11
        DEPOLARIZE2(0.125) 1 2 7 8
        MPP(0.125) Y0 Y1*Y2 Y4 Y5 Y7*Y8 Y10 Y3 Y9 Y6 Y11
        OBSERVABLE_INCLUDE(1) rec[-4] rec[-2]
        DETECTOR(0.5, 3, 0) rec[-40] rec[-36] rec[-30] rec[-16] rec[-8] rec[-4]
        DETECTOR(2.5, 3, 0) rec[-38] rec[-37] rec[-34] rec[-29] rec[-25] rec[-15] rec[-11] rec[-6] rec[-5] rec[-2]
        TICK
        H_YZ 0 1 2 3 4 5 6 7 8 9 10 11
        DEPOLARIZE1(0.125) 0 1 2 3 4 5 6 7 8 9 10 11
        TICK
        X_ERROR(0.125) 0 1 2 3 4 5 6 7 8 9 10 11
        M 0 1 2 3 4 5 6 7 8 9 10 11
        DETECTOR(0, 0.5, 0) rec[-22] rec[-12]
        DETECTOR(1, 0.5, 0) rec[-21] rec[-11] rec[-10]
        DETECTOR(1, 3.5, 0) rec[-20] rec[-8]
        DETECTOR(2, 0.5, 0) rec[-19] rec[-7]
        DETECTOR(2, 3.5, 0) rec[-18] rec[-5] rec[-4]
        DETECTOR(3, 3.5, 0) rec[-17] rec[-2]
        DETECTOR(0.5, 2, 0) rec[-16] rec[-9]
        DETECTOR(1.5, 5, 0) rec[-15] rec[-3]
        DETECTOR(2.5, 2, 0) rec[-14] rec[-6]
        DETECTOR(3.5, 5, 0) rec[-13] rec[-1]
        DETECTOR(1.5, 2, 0) rec[-36] rec[-35] rec[-31] rec[-28] rec[-27] rec[-24] rec[-10] rec[-9] rec[-8] rec[-7] rec[-6] rec[-5]
        OBSERVABLE_INCLUDE(1) rec[-10] rec[-9] rec[-7] rec[-6]
        TICK
    """)
