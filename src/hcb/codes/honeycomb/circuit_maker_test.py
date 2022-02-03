import pytest
import stim

from hcb.codes.honeycomb.layout import HoneycombLayout


@pytest.mark.parametrize("data_width,data_height,rounds,gate_set,tested_observable,decomposed_graphlike_code_distance,ignored_graphlike_code_distance", [
    (8, 12, 10, 'SI1000', 'H', 5, 6),
    (10, 15, 10, 'SI1000', 'H', 7, 7),
    (12, 18, 10, 'SI1000', 'H', 8, 9),
    (14, 21, 10, 'SI1000', 'H', 10, 10),
    (16, 24, 10, 'SI1000', 'H', 11, 12),

    (8, 12, 10, 'SI1000', 'V', 5, 7),
    (10, 15, 10, 'SI1000', 'V', 7, 9),
    (12, 18, 10, 'SI1000', 'V', 9, 11),
    (14, 21, 10, 'SI1000', 'V', 11, 13),
    (16, 24, 10, 'SI1000', 'V', 13, 15),

    (7, 12, 10, 'SI1000', 'EPR', 6, 6),
    (8, 15, 10, 'SI1000', 'EPR', 6, 7),
    (10, 18, 10, 'SI1000', 'EPR', 8, 9),
    (11, 21, 10, 'SI1000', 'EPR', 10, 10),
    (13, 24, 10, 'SI1000', 'EPR', 12, 12),
    (25, 48, 10, 'SI1000', 'EPR', 24, 24),

    (8, 12, 10, 'SD6', 'H', 5, 6),
    (10, 15, 10, 'SD6', 'H', 7, 7),
    (12, 18, 10, 'SD6', 'H', 8, 9),
    (14, 21, 10, 'SD6', 'H', 10, 10),
    (16, 24, 10, 'SD6', 'H', 11, 12),

    (8, 12, 10, 'SD6', 'V', 5, 7),
    (10, 15, 10, 'SD6', 'V', 7, 9),
    (12, 18, 10, 'SD6', 'V', 9, 11),
    (14, 21, 10, 'SD6', 'V', 11, 13),
    (16, 24, 10, 'SD6', 'V', 13, 15),

    (7, 12, 10, 'SD6', 'EPR', 6, 6),
    (8, 15, 10, 'SD6', 'EPR', 6, 7),
    (10, 18, 10, 'SD6', 'EPR', 8, 9),
    (11, 21, 10, 'SD6', 'EPR', 10, 10),
    (13, 24, 10, 'SD6', 'EPR', 12, 12),
    (25, 48, 10, 'SD6', 'EPR', 24, 24),

    (8, 12, 10, 'EM3_v1', 'H', 4, 4),
    (8, 15, 10, 'EM3_v1', 'H', 5, 5),
    (10, 12, 10, 'EM3_v1', 'H', 4, 4),

    (8, 12, 10, 'EM3_v1', 'V', 4, 4),
    (8, 15, 10, 'EM3_v1', 'V', 4, 4),
    (10, 12, 10, 'EM3_v1', 'V', 5, 5),

    (8, 12, 10, 'EM3_v1', 'EPR', 4, 4),
    (8, 15, 10, 'EM3_v1', 'EPR', 4, 4),
    (10, 12, 10, 'EM3_v1', 'EPR', 4, 4),
    (10, 15, 10, 'EM3_v1', 'EPR', 5, 5),

    (8, 12, 10, 'EM3_v2', 'EPR', 4, 4),
])
def test_graphlike_code_distances(*,
                                  data_width: int,
                                  data_height: int,
                                  rounds: int,
                                  gate_set: str,
                                  tested_observable: str,
                                  ignored_graphlike_code_distance: int,
                                  decomposed_graphlike_code_distance: int):
    layout = HoneycombLayout(data_width=data_width,
                             data_height=data_height,
                             rounds=rounds,
                             noise_level=0.001,
                             noisy_gate_set=gate_set,
                             tested_observable=tested_observable)
    circuit = layout.noisy_circuit()

    err = circuit.shortest_graphlike_error(ignore_ungraphlike_errors=True)
    assert len(err) == ignored_graphlike_code_distance, "ignored"

    dem = circuit.detector_error_model(decompose_errors=True)
    err = dem.shortest_graphlike_error(ignore_ungraphlike_errors=False)
    assert len(err) == decomposed_graphlike_code_distance, "decomposed"


def test_exact_circuit_EM3_v1_H():
    layout = HoneycombLayout(data_width=2,
                             data_height=6,
                             rounds=100,
                             noise_level=0.125,
                             noisy_gate_set='EM3_v1',
                             tested_observable='H',
                             sheared=True)
    assert layout.ideal_and_noisy_circuit[1] == stim.Circuit("""
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
        DEPOLARIZE2(0.125) 2 3 6 5 8 9 11 10 0 1 4 7
        MPP(0.125) X2*X3 X6*X5 X8*X9 X11*X10 X0*X1 X4*X7
        SHIFT_COORDS(0, 0, 1)
        TICK
        DEPOLARIZE1(0.125) 0 4 5 10 3 9 6 11
        DEPOLARIZE2(0.125) 2 1 8 7
        MPP(0.125) Y0 Y2*Y1 Y4 Y5 Y8*Y7 Y10 Y3 Y9 Y6 Y11
        OBSERVABLE_INCLUDE(1) rec[-3] rec[-1]
        DETECTOR(1.5, 4, 0) rec[-8] rec[-6] rec[-3]
        DETECTOR(2.5, 1, 0) rec[-7] rec[-2]
        DETECTOR(0.5, 1, 0) rec[-10] rec[-9] rec[-4]
        DETECTOR(3.5, 4, 0) rec[-5] rec[-1]
        SHIFT_COORDS(0, 0, 1)
        TICK
        DEPOLARIZE1(0.125) 0 1 9 11
        DEPOLARIZE2(0.125) 4 3 6 7 2 5 8 10
        MPP(0.125) Z0 Z1 Z4*Z3 Z6*Z7 Z9 Z11 Z2*Z5 Z8*Z10
        OBSERVABLE_INCLUDE(1) rec[-1]
        SHIFT_COORDS(0, 0, 1)
        TICK
        DEPOLARIZE2(0.125) 2 3 6 5 8 9 11 10 0 1 4 7
        MPP(0.125) X2*X3 X6*X5 X8*X9 X11*X10 X0*X1 X4*X7
        DETECTOR(1.5, 2, 0) rec[-12] rec[-11] rec[-8] rec[-6] rec[-5] rec[-1]
        SHIFT_COORDS(0, 0, 1)
        TICK
        DEPOLARIZE1(0.125) 0 1 9 11
        DEPOLARIZE2(0.125) 4 3 6 7 2 5 8 10
        MPP(0.125) Z0 Z1 Z4*Z3 Z6*Z7 Z9 Z11 Z2*Z5 Z8*Z10
        OBSERVABLE_INCLUDE(1) rec[-1]
        DETECTOR(1.5, 2, 0) rec[-20] rec[-19] rec[-16] rec[-6] rec[-5] rec[-2]
        DETECTOR(2.5, 5, 0) rec[-18] rec[-17] rec[-15] rec[-4] rec[-3] rec[-1]
        DETECTOR(0.5, -1, 0) rec[-22] rec[-21] rec[-8] rec[-7]
        SHIFT_COORDS(0, 0, 1)
        TICK
        DEPOLARIZE1(0.125) 0 4 5 10 3 9 6 11
        DEPOLARIZE2(0.125) 2 1 8 7
        MPP(0.125) Y0 Y2*Y1 Y4 Y5 Y8*Y7 Y10 Y3 Y9 Y6 Y11
        OBSERVABLE_INCLUDE(1) rec[-3] rec[-1]
        DETECTOR(0.5, 3, 0) rec[-40] rec[-36] rec[-30] rec[-16] rec[-8] rec[-4]
        DETECTOR(2.5, 3, 0) rec[-38] rec[-37] rec[-34] rec[-29] rec[-25] rec[-15] rec[-11] rec[-6] rec[-5] rec[-2]
        SHIFT_COORDS(0, 0, 1)
        TICK
        REPEAT 48 {
            DEPOLARIZE2(0.125) 2 3 6 5 8 9 11 10 0 1 4 7
            MPP(0.125) X2*X3 X6*X5 X8*X9 X11*X10 X0*X1 X4*X7
            SHIFT_COORDS(0, 0, 1)
            TICK
            DEPOLARIZE1(0.125) 0 4 5 10 3 9 6 11
            DEPOLARIZE2(0.125) 2 1 8 7
            MPP(0.125) Y0 Y2*Y1 Y4 Y5 Y8*Y7 Y10 Y3 Y9 Y6 Y11
            OBSERVABLE_INCLUDE(1) rec[-3] rec[-1]
            DETECTOR(1.5, 4, 0) rec[-24] rec[-22] rec[-19] rec[-8] rec[-6] rec[-3]
            DETECTOR(2.5, 1, 0) rec[-23] rec[-18] rec[-7] rec[-2]
            DETECTOR(0.5, 1, 0) rec[-26] rec[-25] rec[-20] rec[-10] rec[-9] rec[-4]
            DETECTOR(3.5, 4, 0) rec[-21] rec[-17] rec[-5] rec[-1]
            SHIFT_COORDS(0, 0, 1)
            TICK
            DEPOLARIZE1(0.125) 0 1 9 11
            DEPOLARIZE2(0.125) 4 3 6 7 2 5 8 10
            MPP(0.125) Z0 Z1 Z4*Z3 Z6*Z7 Z9 Z11 Z2*Z5 Z8*Z10
            OBSERVABLE_INCLUDE(1) rec[-1]
            SHIFT_COORDS(0, 0, 1)
            TICK
            DEPOLARIZE2(0.125) 2 3 6 5 8 9 11 10 0 1 4 7
            MPP(0.125) X2*X3 X6*X5 X8*X9 X11*X10 X0*X1 X4*X7
            DETECTOR(1.5, 2, 0) rec[-54] rec[-53] rec[-49] rec[-46] rec[-45] rec[-42] rec[-12] rec[-11] rec[-8] rec[-6] rec[-5] rec[-1]
            SHIFT_COORDS(0, 0, 1)
            TICK
            DEPOLARIZE1(0.125) 0 1 9 11
            DEPOLARIZE2(0.125) 4 3 6 7 2 5 8 10
            MPP(0.125) Z0 Z1 Z4*Z3 Z6*Z7 Z9 Z11 Z2*Z5 Z8*Z10
            OBSERVABLE_INCLUDE(1) rec[-1]
            DETECTOR(1.5, 2, 0) rec[-20] rec[-19] rec[-16] rec[-6] rec[-5] rec[-2]
            DETECTOR(2.5, 5, 0) rec[-18] rec[-17] rec[-15] rec[-4] rec[-3] rec[-1]
            DETECTOR(0.5, -1, 0) rec[-22] rec[-21] rec[-8] rec[-7]
            SHIFT_COORDS(0, 0, 1)
            TICK
            DEPOLARIZE1(0.125) 0 4 5 10 3 9 6 11
            DEPOLARIZE2(0.125) 2 1 8 7
            MPP(0.125) Y0 Y2*Y1 Y4 Y5 Y8*Y7 Y10 Y3 Y9 Y6 Y11
            OBSERVABLE_INCLUDE(1) rec[-3] rec[-1]
            DETECTOR(0.5, 3, 0) rec[-40] rec[-36] rec[-30] rec[-16] rec[-8] rec[-4]
            DETECTOR(2.5, 3, 0) rec[-38] rec[-37] rec[-34] rec[-29] rec[-25] rec[-15] rec[-11] rec[-6] rec[-5] rec[-2]
            SHIFT_COORDS(0, 0, 1)
            TICK
        }
        DEPOLARIZE2(0.125) 2 3 6 5 8 9 11 10 0 1 4 7
        MPP(0.125) X2*X3 X6*X5 X8*X9 X11*X10 X0*X1 X4*X7
        SHIFT_COORDS(0, 0, 1)
        TICK
        DEPOLARIZE1(0.125) 0 4 5 10 3 9 6 11
        DEPOLARIZE2(0.125) 2 1 8 7
        MPP(0.125) Y0 Y2*Y1 Y4 Y5 Y8*Y7 Y10 Y3 Y9 Y6 Y11
        OBSERVABLE_INCLUDE(1) rec[-3] rec[-1]
        DETECTOR(1.5, 4, 0) rec[-24] rec[-22] rec[-19] rec[-8] rec[-6] rec[-3]
        DETECTOR(2.5, 1, 0) rec[-23] rec[-18] rec[-7] rec[-2]
        DETECTOR(0.5, 1, 0) rec[-26] rec[-25] rec[-20] rec[-10] rec[-9] rec[-4]
        DETECTOR(3.5, 4, 0) rec[-21] rec[-17] rec[-5] rec[-1]
        SHIFT_COORDS(0, 0, 1)
        TICK
        DEPOLARIZE1(0.125) 0 1 9 11
        DEPOLARIZE2(0.125) 4 3 6 7 2 5 8 10
        MPP(0.125) Z0 Z1 Z4*Z3 Z6*Z7 Z9 Z11 Z2*Z5 Z8*Z10
        OBSERVABLE_INCLUDE(1) rec[-1]
        SHIFT_COORDS(0, 0, 1)
        TICK
        DEPOLARIZE2(0.125) 2 3 6 5 8 9 11 10 0 1 4 7
        MPP(0.125) X2*X3 X6*X5 X8*X9 X11*X10 X0*X1 X4*X7
        DETECTOR(1.5, 2, 0) rec[-54] rec[-53] rec[-49] rec[-46] rec[-45] rec[-42] rec[-12] rec[-11] rec[-8] rec[-6] rec[-5] rec[-1]
        SHIFT_COORDS(0, 0, 1)
        TICK
        DEPOLARIZE1(0.125) 0 1 9 11
        DEPOLARIZE2(0.125) 4 3 6 7 2 5 8 10
        MPP(0.125) Z0 Z1 Z4*Z3 Z6*Z7 Z9 Z11 Z2*Z5 Z8*Z10
        OBSERVABLE_INCLUDE(1) rec[-1]
        DETECTOR(1.5, 2, 0) rec[-20] rec[-19] rec[-16] rec[-6] rec[-5] rec[-2]
        DETECTOR(2.5, 5, 0) rec[-18] rec[-17] rec[-15] rec[-4] rec[-3] rec[-1]
        DETECTOR(0.5, -1, 0) rec[-22] rec[-21] rec[-8] rec[-7]
        SHIFT_COORDS(0, 0, 1)
        TICK
        DEPOLARIZE1(0.125) 0 4 5 10 3 9 6 11
        DEPOLARIZE2(0.125) 2 1 8 7
        MPP(0.125) Y0 Y2*Y1 Y4 Y5 Y8*Y7 Y10 Y3 Y9 Y6 Y11
        OBSERVABLE_INCLUDE(1) rec[-3] rec[-1]
        DETECTOR(0.5, 3, 0) rec[-40] rec[-36] rec[-30] rec[-16] rec[-8] rec[-4]
        DETECTOR(2.5, 3, 0) rec[-38] rec[-37] rec[-34] rec[-29] rec[-25] rec[-15] rec[-11] rec[-6] rec[-5] rec[-2]
        SHIFT_COORDS(0, 0, 1)
        TICK
        DEPOLARIZE1(0.125) 0 1 2 3 4 5 6 7 8 9 10 11
        MPP(0.125) Y0 Y1 Y2 Y3 Y4 Y5 Y6 Y7 Y8 Y9 Y10 Y11
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
        OBSERVABLE_INCLUDE(1) rec[-4] rec[-3] rec[-2] rec[-1]
        TICK
    """)


def test_exact_circuit_SD6_V():
    layout = HoneycombLayout(data_width=2,
                             data_height=6,
                             rounds=100,
                             noise_level=0.125,
                             noisy_gate_set='SD6',
                             tested_observable='V',
                             sheared=False)
    assert layout.ideal_and_noisy_circuit[1] == stim.Circuit("""
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(0, 4) 1
        QUBIT_COORDS(0, 5) 2
        QUBIT_COORDS(1, 0) 3
        QUBIT_COORDS(1, 1) 4
        QUBIT_COORDS(1, 2) 5
        QUBIT_COORDS(1, 3) 6
        QUBIT_COORDS(1, 4) 7
        QUBIT_COORDS(1, 5) 8
        QUBIT_COORDS(2, 1) 9
        QUBIT_COORDS(2, 2) 10
        QUBIT_COORDS(2, 3) 11
        QUBIT_COORDS(0, -0.5) 12
        QUBIT_COORDS(0, 0.5) 13
        QUBIT_COORDS(0, 3.5) 14
        QUBIT_COORDS(0, 4.5) 15
        QUBIT_COORDS(0, 5.5) 16
        QUBIT_COORDS(1, -0.5) 17
        QUBIT_COORDS(1, 0.5) 18
        QUBIT_COORDS(1, 1.5) 19
        QUBIT_COORDS(1, 2.5) 20
        QUBIT_COORDS(1, 3.5) 21
        QUBIT_COORDS(1, 4.5) 22
        QUBIT_COORDS(1, 5.5) 23
        QUBIT_COORDS(2, 0.5) 24
        QUBIT_COORDS(2, 1.5) 25
        QUBIT_COORDS(2, 2.5) 26
        QUBIT_COORDS(2, 3.5) 27
        QUBIT_COORDS(-0.5, 5) 28
        QUBIT_COORDS(0.5, 0) 29
        QUBIT_COORDS(0.5, 2) 30
        QUBIT_COORDS(0.5, 4) 31
        QUBIT_COORDS(1.5, 1) 32
        QUBIT_COORDS(1.5, 3) 33
        QUBIT_COORDS(1.5, 5) 34
        QUBIT_COORDS(2.5, 2) 35
        R 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 0 1 2 3 4 5 6 7 8 9 10 11
        X_ERROR(0.125) 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 0 1 2 3 4 5 6 7 8 9 10 11
        TICK
        CX 1 15 4 19 8 22 10 25 0 29 6 33
        DEPOLARIZE2(0.125) 1 15 4 19 8 22 10 25 0 29 6 33
        DEPOLARIZE1(0.125) 2 3 5 7 9 11 12 13 14 16 17 18 20 21 23 24 26 27 28 30 31 32 34 35
        TICK
        R 13 14 18 21 24 27 28 30 34 35
        CX 2 15 5 19 7 22 9 25 3 29 11 33
        C_ZYX 0 1 4 6 8 10
        X_ERROR(0.125) 13 14 18 21 24 27 28 30 34 35
        DEPOLARIZE2(0.125) 2 15 5 19 7 22 9 25 3 29 11 33
        DEPOLARIZE1(0.125) 0 1 4 6 8 10 12 16 17 20 23 26 31 32
        TICK
        X_ERROR(0.125) 15 19 22 25 29 33
        CX 0 13 1 14 4 18 6 21 8 34 10 35
        C_ZYX 2 3 5 7 9 11
        M 15 19 22 25 29 33
        DETECTOR(0, 4.5, 0) rec[-6]
        DETECTOR(1, 1.5, 0) rec[-5]
        DETECTOR(1, 4.5, 0) rec[-4]
        DETECTOR(2, 1.5, 0) rec[-3]
        DETECTOR(0.5, 0, 0) rec[-2]
        DETECTOR(1.5, 3, 0) rec[-1]
        SHIFT_COORDS(0, 0, 1)
        DEPOLARIZE2(0.125) 0 13 1 14 4 18 6 21 8 34 10 35
        DEPOLARIZE1(0.125) 2 3 5 7 9 11 12 16 17 20 23 24 26 27 28 30 31 32
        TICK
        R 12 16 17 20 23 26 31 32
        CX 3 18 7 21 9 24 11 27 2 28 5 30
        C_ZYX 0 1 4 6 8 10
        X_ERROR(0.125) 12 16 17 20 23 26 31 32
        DEPOLARIZE2(0.125) 3 18 7 21 9 24 11 27 2 28 5 30
        DEPOLARIZE1(0.125) 0 1 4 6 8 10 13 14 15 19 22 25 29 33 34 35
        TICK
        X_ERROR(0.125) 13 14 18 21 24 27 28 30 34 35
        CX 0 12 6 20 8 23 10 26 1 31 4 32
        C_ZYX 2 3 5 7 9 11
        M 13 14 18 21 24 27 28 30 34 35
        OBSERVABLE_INCLUDE(0) rec[-8] rec[-7]
        SHIFT_COORDS(0, 0, 1)
        DEPOLARIZE2(0.125) 0 12 6 20 8 23 10 26 1 31 4 32
        DEPOLARIZE1(0.125) 2 3 5 7 9 11 15 16 17 19 22 25 29 33
        TICK
        R 15 19 22 25 29 33
        CX 2 16 3 17 5 20 11 26 7 31 9 32
        C_ZYX 0 1 4 6 8 10
        X_ERROR(0.125) 15 19 22 25 29 33
        DEPOLARIZE2(0.125) 2 16 3 17 5 20 11 26 7 31 9 32
        DEPOLARIZE1(0.125) 0 1 4 6 8 10 12 13 14 18 21 23 24 27 28 30 34 35
        TICK
        X_ERROR(0.125) 12 16 17 20 23 26 31 32
        CX 1 15 4 19 8 22 10 25 0 29 6 33
        C_ZYX 2 3 5 7 9 11
        M 12 16 17 20 23 26 31 32
        OBSERVABLE_INCLUDE(0) rec[-6] rec[-5] rec[-4]
        SHIFT_COORDS(0, 0, 1)
        DEPOLARIZE2(0.125) 1 15 4 19 8 22 10 25 0 29 6 33
        DEPOLARIZE1(0.125) 2 3 5 7 9 11 13 14 18 21 24 27 28 30 34 35
        TICK
        R 12 16 17 20 23 26 31 32
        CX 2 15 5 19 7 22 9 25 3 29 11 33
        C_XYZ 0 1 4 6 8 10
        X_ERROR(0.125) 12 16 17 20 23 26 31 32
        DEPOLARIZE2(0.125) 2 15 5 19 7 22 9 25 3 29 11 33
        DEPOLARIZE1(0.125) 0 1 4 6 8 10 13 14 18 21 24 27 28 30 34 35
        TICK
        X_ERROR(0.125) 15 19 22 25 29 33
        CX 0 12 6 20 8 23 10 26 1 31 4 32
        C_XYZ 2 3 5 7 9 11
        M 15 19 22 25 29 33
        SHIFT_COORDS(0, 0, 1)
        DEPOLARIZE2(0.125) 0 12 6 20 8 23 10 26 1 31 4 32
        DEPOLARIZE1(0.125) 2 3 5 7 9 11 13 14 16 17 18 21 24 27 28 30 34 35
        TICK
        R 13 14 18 21 24 27 28 30 34 35
        CX 2 16 3 17 5 20 11 26 7 31 9 32
        C_XYZ 0 1 4 6 8 10
        X_ERROR(0.125) 13 14 18 21 24 27 28 30 34 35
        DEPOLARIZE2(0.125) 2 16 3 17 5 20 11 26 7 31 9 32
        DEPOLARIZE1(0.125) 0 1 4 6 8 10 12 15 19 22 23 25 29 33
        TICK
        X_ERROR(0.125) 12 16 17 20 23 26 31 32
        CX 0 13 1 14 4 18 6 21 8 34 10 35
        C_XYZ 2 3 5 7 9 11
        M 12 16 17 20 23 26 31 32
        OBSERVABLE_INCLUDE(0) rec[-6] rec[-5] rec[-4]
        DETECTOR(1.5, 2, 0) rec[-19] rec[-17] rec[-15] rec[-5] rec[-3] rec[-1]
        DETECTOR(0.5, 5, 0) rec[-21] rec[-18] rec[-16] rec[-7] rec[-4] rec[-2]
        DETECTOR(0.5, -1, 0) rec[-22] rec[-20] rec[-8] rec[-6]
        SHIFT_COORDS(0, 0, 1)
        DEPOLARIZE2(0.125) 0 13 1 14 4 18 6 21 8 34 10 35
        DEPOLARIZE1(0.125) 2 3 5 7 9 11 15 19 22 24 25 27 28 29 30 33
        TICK
        R 15 19 22 25 29 33
        CX 3 18 7 21 9 24 11 27 2 28 5 30
        C_XYZ 0 1 4 6 8 10
        X_ERROR(0.125) 15 19 22 25 29 33
        DEPOLARIZE2(0.125) 3 18 7 21 9 24 11 27 2 28 5 30
        DEPOLARIZE1(0.125) 0 1 4 6 8 10 12 13 14 16 17 20 23 26 31 32 34 35
        TICK
        X_ERROR(0.125) 13 14 18 21 24 27 28 30 34 35
        CX 1 15 4 19 8 22 10 25 0 29 6 33
        C_XYZ 2 3 5 7 9 11
        M 13 14 18 21 24 27 28 30 34 35
        OBSERVABLE_INCLUDE(0) rec[-8] rec[-7]
        DETECTOR(2.5, 3, 0) rec[-37] rec[-33] rec[-27] rec[-13] rec[-5] rec[-1]
        DETECTOR(0.5, 3, 0) rec[-41] rec[-39] rec[-35] rec[-29] rec[-26] rec[-15] rec[-12] rec[-9] rec[-7] rec[-3]
        SHIFT_COORDS(0, 0, 1)
        DEPOLARIZE2(0.125) 1 15 4 19 8 22 10 25 0 29 6 33
        DEPOLARIZE1(0.125) 2 3 5 7 9 11 12 16 17 20 23 26 31 32
        TICK
        REPEAT 48 {
            R 13 14 18 21 24 27 28 30 34 35
            CX 2 15 5 19 7 22 9 25 3 29 11 33
            C_ZYX 0 1 4 6 8 10
            X_ERROR(0.125) 13 14 18 21 24 27 28 30 34 35
            DEPOLARIZE2(0.125) 2 15 5 19 7 22 9 25 3 29 11 33
            DEPOLARIZE1(0.125) 0 1 4 6 8 10 12 16 17 20 23 26 31 32
            TICK
            X_ERROR(0.125) 15 19 22 25 29 33
            CX 0 13 1 14 4 18 6 21 8 34 10 35
            C_ZYX 2 3 5 7 9 11
            M 15 19 22 25 29 33
            SHIFT_COORDS(0, 0, 1)
            DEPOLARIZE2(0.125) 0 13 1 14 4 18 6 21 8 34 10 35
            DEPOLARIZE1(0.125) 2 3 5 7 9 11 12 16 17 20 23 24 26 27 28 30 31 32
            TICK
            R 12 16 17 20 23 26 31 32
            CX 3 18 7 21 9 24 11 27 2 28 5 30
            C_ZYX 0 1 4 6 8 10
            X_ERROR(0.125) 12 16 17 20 23 26 31 32
            DEPOLARIZE2(0.125) 3 18 7 21 9 24 11 27 2 28 5 30
            DEPOLARIZE1(0.125) 0 1 4 6 8 10 13 14 15 19 22 25 29 33 34 35
            TICK
            X_ERROR(0.125) 13 14 18 21 24 27 28 30 34 35
            CX 0 12 6 20 8 23 10 26 1 31 4 32
            C_ZYX 2 3 5 7 9 11
            M 13 14 18 21 24 27 28 30 34 35
            OBSERVABLE_INCLUDE(0) rec[-8] rec[-7]
            DETECTOR(2.5, 1, 0) rec[-22] rec[-17] rec[-6] rec[-1]
            DETECTOR(1.5, 4, 0) rec[-23] rec[-21] rec[-18] rec[-7] rec[-5] rec[-2]
            DETECTOR(-0.5, 4, 0) rec[-25] rec[-20] rec[-9] rec[-4]
            DETECTOR(0.5, 1, 0) rec[-26] rec[-24] rec[-19] rec[-10] rec[-8] rec[-3]
            SHIFT_COORDS(0, 0, 1)
            DEPOLARIZE2(0.125) 0 12 6 20 8 23 10 26 1 31 4 32
            DEPOLARIZE1(0.125) 2 3 5 7 9 11 15 16 17 19 22 25 29 33
            TICK
            R 15 19 22 25 29 33
            CX 2 16 3 17 5 20 11 26 7 31 9 32
            C_ZYX 0 1 4 6 8 10
            X_ERROR(0.125) 15 19 22 25 29 33
            DEPOLARIZE2(0.125) 2 16 3 17 5 20 11 26 7 31 9 32
            DEPOLARIZE1(0.125) 0 1 4 6 8 10 12 13 14 18 21 23 24 27 28 30 34 35
            TICK
            X_ERROR(0.125) 12 16 17 20 23 26 31 32
            CX 1 15 4 19 8 22 10 25 0 29 6 33
            C_ZYX 2 3 5 7 9 11
            M 12 16 17 20 23 26 31 32
            OBSERVABLE_INCLUDE(0) rec[-6] rec[-5] rec[-4]
            SHIFT_COORDS(0, 0, 1)
            DEPOLARIZE2(0.125) 1 15 4 19 8 22 10 25 0 29 6 33
            DEPOLARIZE1(0.125) 2 3 5 7 9 11 13 14 18 21 24 27 28 30 34 35
            TICK
            R 12 16 17 20 23 26 31 32
            CX 2 15 5 19 7 22 9 25 3 29 11 33
            C_XYZ 0 1 4 6 8 10
            X_ERROR(0.125) 12 16 17 20 23 26 31 32
            DEPOLARIZE2(0.125) 2 15 5 19 7 22 9 25 3 29 11 33
            DEPOLARIZE1(0.125) 0 1 4 6 8 10 13 14 18 21 24 27 28 30 34 35
            TICK
            X_ERROR(0.125) 15 19 22 25 29 33
            CX 0 12 6 20 8 23 10 26 1 31 4 32
            C_XYZ 2 3 5 7 9 11
            M 15 19 22 25 29 33
            DETECTOR(1.5, 2, 0) rec[-53] rec[-51] rec[-49] rec[-45] rec[-43] rec[-41] rec[-11] rec[-9] rec[-7] rec[-5] rec[-3] rec[-1]
            SHIFT_COORDS(0, 0, 1)
            DEPOLARIZE2(0.125) 0 12 6 20 8 23 10 26 1 31 4 32
            DEPOLARIZE1(0.125) 2 3 5 7 9 11 13 14 16 17 18 21 24 27 28 30 34 35
            TICK
            R 13 14 18 21 24 27 28 30 34 35
            CX 2 16 3 17 5 20 11 26 7 31 9 32
            C_XYZ 0 1 4 6 8 10
            X_ERROR(0.125) 13 14 18 21 24 27 28 30 34 35
            DEPOLARIZE2(0.125) 2 16 3 17 5 20 11 26 7 31 9 32
            DEPOLARIZE1(0.125) 0 1 4 6 8 10 12 15 19 22 23 25 29 33
            TICK
            X_ERROR(0.125) 12 16 17 20 23 26 31 32
            CX 0 13 1 14 4 18 6 21 8 34 10 35
            C_XYZ 2 3 5 7 9 11
            M 12 16 17 20 23 26 31 32
            OBSERVABLE_INCLUDE(0) rec[-6] rec[-5] rec[-4]
            DETECTOR(1.5, 2, 0) rec[-19] rec[-17] rec[-15] rec[-5] rec[-3] rec[-1]
            DETECTOR(0.5, 5, 0) rec[-21] rec[-18] rec[-16] rec[-7] rec[-4] rec[-2]
            DETECTOR(0.5, -1, 0) rec[-22] rec[-20] rec[-8] rec[-6]
            SHIFT_COORDS(0, 0, 1)
            DEPOLARIZE2(0.125) 0 13 1 14 4 18 6 21 8 34 10 35
            DEPOLARIZE1(0.125) 2 3 5 7 9 11 15 19 22 24 25 27 28 29 30 33
            TICK
            R 15 19 22 25 29 33
            CX 3 18 7 21 9 24 11 27 2 28 5 30
            C_XYZ 0 1 4 6 8 10
            X_ERROR(0.125) 15 19 22 25 29 33
            DEPOLARIZE2(0.125) 3 18 7 21 9 24 11 27 2 28 5 30
            DEPOLARIZE1(0.125) 0 1 4 6 8 10 12 13 14 16 17 20 23 26 31 32 34 35
            TICK
            X_ERROR(0.125) 13 14 18 21 24 27 28 30 34 35
            CX 1 15 4 19 8 22 10 25 0 29 6 33
            C_XYZ 2 3 5 7 9 11
            M 13 14 18 21 24 27 28 30 34 35
            OBSERVABLE_INCLUDE(0) rec[-8] rec[-7]
            DETECTOR(2.5, 3, 0) rec[-37] rec[-33] rec[-27] rec[-13] rec[-5] rec[-1]
            DETECTOR(0.5, 3, 0) rec[-41] rec[-39] rec[-35] rec[-29] rec[-26] rec[-15] rec[-12] rec[-9] rec[-7] rec[-3]
            SHIFT_COORDS(0, 0, 1)
            DEPOLARIZE2(0.125) 1 15 4 19 8 22 10 25 0 29 6 33
            DEPOLARIZE1(0.125) 2 3 5 7 9 11 12 16 17 20 23 26 31 32
            TICK
        }
        R 13 14 18 21 24 27 28 30 34 35
        CX 2 15 5 19 7 22 9 25 3 29 11 33
        C_ZYX 0 1 4 6 8 10
        X_ERROR(0.125) 13 14 18 21 24 27 28 30 34 35
        DEPOLARIZE2(0.125) 2 15 5 19 7 22 9 25 3 29 11 33
        DEPOLARIZE1(0.125) 0 1 4 6 8 10 12 16 17 20 23 26 31 32
        TICK
        X_ERROR(0.125) 15 19 22 25 29 33
        CX 0 13 1 14 4 18 6 21 8 34 10 35
        C_ZYX 2 3 5 7 9 11
        M 15 19 22 25 29 33
        SHIFT_COORDS(0, 0, 1)
        DEPOLARIZE2(0.125) 0 13 1 14 4 18 6 21 8 34 10 35
        DEPOLARIZE1(0.125) 2 3 5 7 9 11 12 16 17 20 23 24 26 27 28 30 31 32
        TICK
        R 12 16 17 20 23 26 31 32
        CX 3 18 7 21 9 24 11 27 2 28 5 30
        C_ZYX 0 1 4 6 8 10
        X_ERROR(0.125) 12 16 17 20 23 26 31 32
        DEPOLARIZE2(0.125) 3 18 7 21 9 24 11 27 2 28 5 30
        DEPOLARIZE1(0.125) 0 1 4 6 8 10 13 14 15 19 22 25 29 33 34 35
        TICK
        X_ERROR(0.125) 13 14 18 21 24 27 28 30 34 35
        CX 0 12 6 20 8 23 10 26 1 31 4 32
        C_ZYX 2 3 5 7 9 11
        M 13 14 18 21 24 27 28 30 34 35
        OBSERVABLE_INCLUDE(0) rec[-8] rec[-7]
        DETECTOR(2.5, 1, 0) rec[-22] rec[-17] rec[-6] rec[-1]
        DETECTOR(1.5, 4, 0) rec[-23] rec[-21] rec[-18] rec[-7] rec[-5] rec[-2]
        DETECTOR(-0.5, 4, 0) rec[-25] rec[-20] rec[-9] rec[-4]
        DETECTOR(0.5, 1, 0) rec[-26] rec[-24] rec[-19] rec[-10] rec[-8] rec[-3]
        SHIFT_COORDS(0, 0, 1)
        DEPOLARIZE2(0.125) 0 12 6 20 8 23 10 26 1 31 4 32
        DEPOLARIZE1(0.125) 2 3 5 7 9 11 15 16 17 19 22 25 29 33
        TICK
        R 15 19 22 25 29 33
        CX 2 16 3 17 5 20 11 26 7 31 9 32
        C_ZYX 0 1 4 6 8 10
        X_ERROR(0.125) 15 19 22 25 29 33
        DEPOLARIZE2(0.125) 2 16 3 17 5 20 11 26 7 31 9 32
        DEPOLARIZE1(0.125) 0 1 4 6 8 10 12 13 14 18 21 23 24 27 28 30 34 35
        TICK
        X_ERROR(0.125) 12 16 17 20 23 26 31 32
        CX 1 15 4 19 8 22 10 25 0 29 6 33
        C_ZYX 2 3 5 7 9 11
        M 12 16 17 20 23 26 31 32
        OBSERVABLE_INCLUDE(0) rec[-6] rec[-5] rec[-4]
        SHIFT_COORDS(0, 0, 1)
        DEPOLARIZE2(0.125) 1 15 4 19 8 22 10 25 0 29 6 33
        DEPOLARIZE1(0.125) 2 3 5 7 9 11 13 14 18 21 24 27 28 30 34 35
        TICK
        R 12 16 17 20 23 26 31 32
        CX 2 15 5 19 7 22 9 25 3 29 11 33
        C_XYZ 0 1 4 6 8 10
        X_ERROR(0.125) 12 16 17 20 23 26 31 32
        DEPOLARIZE2(0.125) 2 15 5 19 7 22 9 25 3 29 11 33
        DEPOLARIZE1(0.125) 0 1 4 6 8 10 13 14 18 21 24 27 28 30 34 35
        TICK
        X_ERROR(0.125) 15 19 22 25 29 33
        CX 0 12 6 20 8 23 10 26 1 31 4 32
        C_XYZ 2 3 5 7 9 11
        M 15 19 22 25 29 33
        DETECTOR(1.5, 2, 0) rec[-53] rec[-51] rec[-49] rec[-45] rec[-43] rec[-41] rec[-11] rec[-9] rec[-7] rec[-5] rec[-3] rec[-1]
        SHIFT_COORDS(0, 0, 1)
        DEPOLARIZE2(0.125) 0 12 6 20 8 23 10 26 1 31 4 32
        DEPOLARIZE1(0.125) 2 3 5 7 9 11 13 14 16 17 18 21 24 27 28 30 34 35
        TICK
        R 13 14 18 21 24 27 28 30 34 35
        CX 2 16 3 17 5 20 11 26 7 31 9 32
        C_XYZ 0 1 4 6 8 10
        X_ERROR(0.125) 13 14 18 21 24 27 28 30 34 35
        DEPOLARIZE2(0.125) 2 16 3 17 5 20 11 26 7 31 9 32
        DEPOLARIZE1(0.125) 0 1 4 6 8 10 12 15 19 22 23 25 29 33
        TICK
        X_ERROR(0.125) 12 16 17 20 23 26 31 32
        CX 0 13 1 14 4 18 6 21 8 34 10 35
        C_XYZ 2 3 5 7 9 11
        M 12 16 17 20 23 26 31 32
        OBSERVABLE_INCLUDE(0) rec[-6] rec[-5] rec[-4]
        DETECTOR(1.5, 2, 0) rec[-19] rec[-17] rec[-15] rec[-5] rec[-3] rec[-1]
        DETECTOR(0.5, 5, 0) rec[-21] rec[-18] rec[-16] rec[-7] rec[-4] rec[-2]
        DETECTOR(0.5, -1, 0) rec[-22] rec[-20] rec[-8] rec[-6]
        SHIFT_COORDS(0, 0, 1)
        DEPOLARIZE2(0.125) 0 13 1 14 4 18 6 21 8 34 10 35
        DEPOLARIZE1(0.125) 2 3 5 7 9 11 15 19 22 24 25 27 28 29 30 33
        TICK
        R 15 19 22 25 29 33
        CX 3 18 7 21 9 24 11 27 2 28 5 30
        C_XYZ 0 1 4 6 8 10
        X_ERROR(0.125) 15 19 22 25 29 33
        DEPOLARIZE2(0.125) 3 18 7 21 9 24 11 27 2 28 5 30
        DEPOLARIZE1(0.125) 0 1 4 6 8 10 12 13 14 16 17 20 23 26 31 32 34 35
        TICK
        X_ERROR(0.125) 13 14 18 21 24 27 28 30 34 35
        CX 1 15 4 19 8 22 10 25 0 29 6 33
        C_XYZ 2 3 5 7 9 11
        M 13 14 18 21 24 27 28 30 34 35
        OBSERVABLE_INCLUDE(0) rec[-8] rec[-7]
        DETECTOR(2.5, 3, 0) rec[-37] rec[-33] rec[-27] rec[-13] rec[-5] rec[-1]
        DETECTOR(0.5, 3, 0) rec[-41] rec[-39] rec[-35] rec[-29] rec[-26] rec[-15] rec[-12] rec[-9] rec[-7] rec[-3]
        SHIFT_COORDS(0, 0, 1)
        DEPOLARIZE2(0.125) 1 15 4 19 8 22 10 25 0 29 6 33
        DEPOLARIZE1(0.125) 2 3 5 7 9 11 12 16 17 20 23 26 31 32
        TICK
        CX 1 15 4 19 8 22 10 25 0 29 6 33
        DEPOLARIZE2(0.125) 1 15 4 19 8 22 10 25 0 29 6 33
        DEPOLARIZE1(0.125) 2 3 5 7 9 11 12 13 14 16 17 18 20 21 23 24 26 27 28 30 31 32 34 35
        TICK
        X_ERROR(0.125) 0 1 2 3 4 5 6 7 8 9 10 11
        M 0 1 2 3 4 5 6 7 8 9 10 11
        OBSERVABLE_INCLUDE(0) rec[-9] rec[-8] rec[-6] rec[-5]
        DEPOLARIZE1(0.125) 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35
        TICK
    """)


def test_exact_circuit_SI1000_V():
    layout = HoneycombLayout(data_width=2,
                             data_height=6,
                             rounds=100,
                             noise_level=0.125,
                             noisy_gate_set='SI1000',
                             tested_observable='H',
                             sheared=False)
    assert layout.ideal_and_noisy_circuit[1] == stim.Circuit("""
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(0, 4) 1
        QUBIT_COORDS(0, 5) 2
        QUBIT_COORDS(1, 0) 3
        QUBIT_COORDS(1, 1) 4
        QUBIT_COORDS(1, 2) 5
        QUBIT_COORDS(1, 3) 6
        QUBIT_COORDS(1, 4) 7
        QUBIT_COORDS(1, 5) 8
        QUBIT_COORDS(2, 1) 9
        QUBIT_COORDS(2, 2) 10
        QUBIT_COORDS(2, 3) 11
        QUBIT_COORDS(0, -0.5) 12
        QUBIT_COORDS(0, 0.5) 13
        QUBIT_COORDS(0, 3.5) 14
        QUBIT_COORDS(0, 4.5) 15
        QUBIT_COORDS(0, 5.5) 16
        QUBIT_COORDS(1, -0.5) 17
        QUBIT_COORDS(1, 0.5) 18
        QUBIT_COORDS(1, 1.5) 19
        QUBIT_COORDS(1, 2.5) 20
        QUBIT_COORDS(1, 3.5) 21
        QUBIT_COORDS(1, 4.5) 22
        QUBIT_COORDS(1, 5.5) 23
        QUBIT_COORDS(2, 0.5) 24
        QUBIT_COORDS(2, 1.5) 25
        QUBIT_COORDS(2, 2.5) 26
        QUBIT_COORDS(2, 3.5) 27
        QUBIT_COORDS(-0.5, 5) 28
        QUBIT_COORDS(0.5, 0) 29
        QUBIT_COORDS(0.5, 2) 30
        QUBIT_COORDS(0.5, 4) 31
        QUBIT_COORDS(1.5, 1) 32
        QUBIT_COORDS(1.5, 3) 33
        QUBIT_COORDS(1.5, 5) 34
        QUBIT_COORDS(2.5, 2) 35
        R 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35
        X_ERROR(0.25) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35
        TICK
        H 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35
        C_XYZ 0 1 2 3 4 5 6 7 8 9 10 11
        DEPOLARIZE1(0.0125) 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 0 1 2 3 4 5 6 7 8 9 10 11
        TICK
        CZ 1 15 4 19 8 22 10 25 0 29 6 33
        DEPOLARIZE2(0.125) 1 15 4 19 8 22 10 25 0 29 6 33
        DEPOLARIZE1(0.0125) 2 3 5 7 9 11 12 13 14 16 17 18 20 21 23 24 26 27 28 30 31 32 34 35
        TICK
        CZ 2 15 5 19 7 22 9 25 3 29 11 33
        C_ZYX 0 1 4 6 8 10
        DEPOLARIZE2(0.125) 2 15 5 19 7 22 9 25 3 29 11 33
        DEPOLARIZE1(0.0125) 0 1 4 6 8 10 12 13 14 16 17 18 20 21 23 24 26 27 28 30 31 32 34 35
        TICK
        CZ 0 13 1 14 4 18 6 21 8 34 10 35
        C_ZYX 2 3 5 7 9 11
        DEPOLARIZE2(0.125) 0 13 1 14 4 18 6 21 8 34 10 35
        DEPOLARIZE1(0.0125) 2 3 5 7 9 11 12 15 16 17 19 20 22 23 24 25 26 27 28 29 30 31 32 33
        TICK
        CZ 3 18 7 21 9 24 11 27 2 28 5 30
        C_ZYX 0 1 4 6 8 10
        DEPOLARIZE2(0.125) 3 18 7 21 9 24 11 27 2 28 5 30
        DEPOLARIZE1(0.0125) 0 1 4 6 8 10 12 13 14 15 16 17 19 20 22 23 25 26 29 31 32 33 34 35
        TICK
        CZ 0 12 6 20 8 23 10 26 1 31 4 32
        C_ZYX 2 3 5 7 9 11
        DEPOLARIZE2(0.125) 0 12 6 20 8 23 10 26 1 31 4 32
        DEPOLARIZE1(0.0125) 2 3 5 7 9 11 13 14 15 16 17 18 19 21 22 24 25 27 28 29 30 33 34 35
        TICK
        CZ 2 16 3 17 5 20 11 26 7 31 9 32
        DEPOLARIZE2(0.125) 2 16 3 17 5 20 11 26 7 31 9 32
        DEPOLARIZE1(0.0125) 0 1 4 6 8 10 12 13 14 15 18 19 21 22 23 24 25 27 28 29 30 33 34 35
        TICK
        H 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35
        DEPOLARIZE1(0.0125) 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 0 1 2 3 4 5 6 7 8 9 10 11
        TICK
        X_ERROR(0.625) 15 19 22 25 29 33 13 14 18 21 24 27 28 30 34 35 12 16 17 20 23 26 31 32
        M 15 19 22 25 29 33
        SHIFT_COORDS(0, 0, 1)
        M 13 14 18 21 24 27 28 30 34 35
        OBSERVABLE_INCLUDE(1) rec[-4] rec[-2]
        DETECTOR(2.5, 1, 0) rec[-6] rec[-1]
        DETECTOR(1.5, 4, 0) rec[-7] rec[-5] rec[-2]
        DETECTOR(-0.5, 4, 0) rec[-9] rec[-4]
        DETECTOR(0.5, 1, 0) rec[-10] rec[-8] rec[-3]
        SHIFT_COORDS(0, 0, 1)
        M 12 16 17 20 23 26 31 32
        OBSERVABLE_INCLUDE(1) rec[-2]
        SHIFT_COORDS(0, 0, 1)
        DEPOLARIZE1(0.0125) 0 1 2 3 4 5 6 7 8 9 10 11
        DEPOLARIZE1(0.25) 0 1 2 3 4 5 6 7 8 9 10 11
        TICK
        R 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35
        X_ERROR(0.25) 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35
        DEPOLARIZE1(0.0125) 0 1 2 3 4 5 6 7 8 9 10 11
        DEPOLARIZE1(0.25) 0 1 2 3 4 5 6 7 8 9 10 11
        TICK
        H 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35
        C_ZYX 0 1 2 3 4 5 6 7 8 9 10 11
        DEPOLARIZE1(0.0125) 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 0 1 2 3 4 5 6 7 8 9 10 11
        TICK
        CZ 1 15 4 19 8 22 10 25 0 29 6 33
        DEPOLARIZE2(0.125) 1 15 4 19 8 22 10 25 0 29 6 33
        DEPOLARIZE1(0.0125) 2 3 5 7 9 11 12 13 14 16 17 18 20 21 23 24 26 27 28 30 31 32 34 35
        TICK
        CZ 2 15 5 19 7 22 9 25 3 29 11 33
        C_XYZ 0 1 4 6 8 10
        DEPOLARIZE2(0.125) 2 15 5 19 7 22 9 25 3 29 11 33
        DEPOLARIZE1(0.0125) 0 1 4 6 8 10 12 13 14 16 17 18 20 21 23 24 26 27 28 30 31 32 34 35
        TICK
        CZ 0 12 6 20 8 23 10 26 1 31 4 32
        C_XYZ 2 3 5 7 9 11
        DEPOLARIZE2(0.125) 0 12 6 20 8 23 10 26 1 31 4 32
        DEPOLARIZE1(0.0125) 2 3 5 7 9 11 13 14 15 16 17 18 19 21 22 24 25 27 28 29 30 33 34 35
        TICK
        CZ 2 16 3 17 5 20 11 26 7 31 9 32
        C_XYZ 0 1 4 6 8 10
        DEPOLARIZE2(0.125) 2 16 3 17 5 20 11 26 7 31 9 32
        DEPOLARIZE1(0.0125) 0 1 4 6 8 10 12 13 14 15 18 19 21 22 23 24 25 27 28 29 30 33 34 35
        TICK
        CZ 0 13 1 14 4 18 6 21 8 34 10 35
        C_XYZ 2 3 5 7 9 11
        DEPOLARIZE2(0.125) 0 13 1 14 4 18 6 21 8 34 10 35
        DEPOLARIZE1(0.0125) 2 3 5 7 9 11 12 15 16 17 19 20 22 23 24 25 26 27 28 29 30 31 32 33
        TICK
        CZ 3 18 7 21 9 24 11 27 2 28 5 30
        DEPOLARIZE2(0.125) 3 18 7 21 9 24 11 27 2 28 5 30
        DEPOLARIZE1(0.0125) 0 1 4 6 8 10 12 13 14 15 16 17 19 20 22 23 25 26 29 31 32 33 34 35
        TICK
        H 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35
        C_ZYX 0 1 2 3 4 5 6 7 8 9 10 11
        DEPOLARIZE1(0.0125) 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 0 1 2 3 4 5 6 7 8 9 10 11
        TICK
        X_ERROR(0.625) 15 19 22 25 29 33 12 16 17 20 23 26 31 32 13 14 18 21 24 27 28 30 34 35
        M 15 19 22 25 29 33
        DETECTOR(1.5, 2, 0) rec[-11] rec[-9] rec[-7] rec[-5] rec[-3] rec[-1]
        SHIFT_COORDS(0, 0, 1)
        M 12 16 17 20 23 26 31 32
        OBSERVABLE_INCLUDE(1) rec[-2]
        DETECTOR(1.5, 2, 0) rec[-19] rec[-17] rec[-15] rec[-5] rec[-3] rec[-1]
        DETECTOR(0.5, 5, 0) rec[-21] rec[-18] rec[-16] rec[-7] rec[-4] rec[-2]
        DETECTOR(0.5, -1, 0) rec[-22] rec[-20] rec[-8] rec[-6]
        SHIFT_COORDS(0, 0, 1)
        M 13 14 18 21 24 27 28 30 34 35
        OBSERVABLE_INCLUDE(1) rec[-4] rec[-2]
        DETECTOR(2.5, 3, 0) rec[-37] rec[-33] rec[-27] rec[-13] rec[-5] rec[-1]
        DETECTOR(0.5, 3, 0) rec[-41] rec[-39] rec[-35] rec[-29] rec[-26] rec[-15] rec[-12] rec[-9] rec[-7] rec[-3]
        SHIFT_COORDS(0, 0, 1)
        DEPOLARIZE1(0.0125) 0 1 2 3 4 5 6 7 8 9 10 11
        DEPOLARIZE1(0.25) 0 1 2 3 4 5 6 7 8 9 10 11
        TICK
        REPEAT 48 {
            R 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35
            X_ERROR(0.25) 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35
            DEPOLARIZE1(0.0125) 0 1 2 3 4 5 6 7 8 9 10 11
            DEPOLARIZE1(0.25) 0 1 2 3 4 5 6 7 8 9 10 11
            TICK
            H 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35
            C_ZYX 0 1 2 3 4 5 6 7 8 9 10 11
            DEPOLARIZE1(0.0125) 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 0 1 2 3 4 5 6 7 8 9 10 11
            TICK
            CZ 1 15 4 19 8 22 10 25 0 29 6 33
            DEPOLARIZE2(0.125) 1 15 4 19 8 22 10 25 0 29 6 33
            DEPOLARIZE1(0.0125) 2 3 5 7 9 11 12 13 14 16 17 18 20 21 23 24 26 27 28 30 31 32 34 35
            TICK
            CZ 2 15 5 19 7 22 9 25 3 29 11 33
            C_ZYX 0 1 4 6 8 10
            DEPOLARIZE2(0.125) 2 15 5 19 7 22 9 25 3 29 11 33
            DEPOLARIZE1(0.0125) 0 1 4 6 8 10 12 13 14 16 17 18 20 21 23 24 26 27 28 30 31 32 34 35
            TICK
            CZ 0 13 1 14 4 18 6 21 8 34 10 35
            C_ZYX 2 3 5 7 9 11
            DEPOLARIZE2(0.125) 0 13 1 14 4 18 6 21 8 34 10 35
            DEPOLARIZE1(0.0125) 2 3 5 7 9 11 12 15 16 17 19 20 22 23 24 25 26 27 28 29 30 31 32 33
            TICK
            CZ 3 18 7 21 9 24 11 27 2 28 5 30
            C_ZYX 0 1 4 6 8 10
            DEPOLARIZE2(0.125) 3 18 7 21 9 24 11 27 2 28 5 30
            DEPOLARIZE1(0.0125) 0 1 4 6 8 10 12 13 14 15 16 17 19 20 22 23 25 26 29 31 32 33 34 35
            TICK
            CZ 0 12 6 20 8 23 10 26 1 31 4 32
            C_ZYX 2 3 5 7 9 11
            DEPOLARIZE2(0.125) 0 12 6 20 8 23 10 26 1 31 4 32
            DEPOLARIZE1(0.0125) 2 3 5 7 9 11 13 14 15 16 17 18 19 21 22 24 25 27 28 29 30 33 34 35
            TICK
            CZ 2 16 3 17 5 20 11 26 7 31 9 32
            DEPOLARIZE2(0.125) 2 16 3 17 5 20 11 26 7 31 9 32
            DEPOLARIZE1(0.0125) 0 1 4 6 8 10 12 13 14 15 18 19 21 22 23 24 25 27 28 29 30 33 34 35
            TICK
            H 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35
            DEPOLARIZE1(0.0125) 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 0 1 2 3 4 5 6 7 8 9 10 11
            TICK
            X_ERROR(0.625) 15 19 22 25 29 33 13 14 18 21 24 27 28 30 34 35 12 16 17 20 23 26 31 32
            M 15 19 22 25 29 33
            SHIFT_COORDS(0, 0, 1)
            M 13 14 18 21 24 27 28 30 34 35
            OBSERVABLE_INCLUDE(1) rec[-4] rec[-2]
            DETECTOR(2.5, 1, 0) rec[-22] rec[-17] rec[-6] rec[-1]
            DETECTOR(1.5, 4, 0) rec[-23] rec[-21] rec[-18] rec[-7] rec[-5] rec[-2]
            DETECTOR(-0.5, 4, 0) rec[-25] rec[-20] rec[-9] rec[-4]
            DETECTOR(0.5, 1, 0) rec[-26] rec[-24] rec[-19] rec[-10] rec[-8] rec[-3]
            SHIFT_COORDS(0, 0, 1)
            M 12 16 17 20 23 26 31 32
            OBSERVABLE_INCLUDE(1) rec[-2]
            SHIFT_COORDS(0, 0, 1)
            DEPOLARIZE1(0.0125) 0 1 2 3 4 5 6 7 8 9 10 11
            DEPOLARIZE1(0.25) 0 1 2 3 4 5 6 7 8 9 10 11
            TICK
            R 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35
            X_ERROR(0.25) 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35
            DEPOLARIZE1(0.0125) 0 1 2 3 4 5 6 7 8 9 10 11
            DEPOLARIZE1(0.25) 0 1 2 3 4 5 6 7 8 9 10 11
            TICK
            H 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35
            C_ZYX 0 1 2 3 4 5 6 7 8 9 10 11
            DEPOLARIZE1(0.0125) 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 0 1 2 3 4 5 6 7 8 9 10 11
            TICK
            CZ 1 15 4 19 8 22 10 25 0 29 6 33
            DEPOLARIZE2(0.125) 1 15 4 19 8 22 10 25 0 29 6 33
            DEPOLARIZE1(0.0125) 2 3 5 7 9 11 12 13 14 16 17 18 20 21 23 24 26 27 28 30 31 32 34 35
            TICK
            CZ 2 15 5 19 7 22 9 25 3 29 11 33
            C_XYZ 0 1 4 6 8 10
            DEPOLARIZE2(0.125) 2 15 5 19 7 22 9 25 3 29 11 33
            DEPOLARIZE1(0.0125) 0 1 4 6 8 10 12 13 14 16 17 18 20 21 23 24 26 27 28 30 31 32 34 35
            TICK
            CZ 0 12 6 20 8 23 10 26 1 31 4 32
            C_XYZ 2 3 5 7 9 11
            DEPOLARIZE2(0.125) 0 12 6 20 8 23 10 26 1 31 4 32
            DEPOLARIZE1(0.0125) 2 3 5 7 9 11 13 14 15 16 17 18 19 21 22 24 25 27 28 29 30 33 34 35
            TICK
            CZ 2 16 3 17 5 20 11 26 7 31 9 32
            C_XYZ 0 1 4 6 8 10
            DEPOLARIZE2(0.125) 2 16 3 17 5 20 11 26 7 31 9 32
            DEPOLARIZE1(0.0125) 0 1 4 6 8 10 12 13 14 15 18 19 21 22 23 24 25 27 28 29 30 33 34 35
            TICK
            CZ 0 13 1 14 4 18 6 21 8 34 10 35
            C_XYZ 2 3 5 7 9 11
            DEPOLARIZE2(0.125) 0 13 1 14 4 18 6 21 8 34 10 35
            DEPOLARIZE1(0.0125) 2 3 5 7 9 11 12 15 16 17 19 20 22 23 24 25 26 27 28 29 30 31 32 33
            TICK
            CZ 3 18 7 21 9 24 11 27 2 28 5 30
            DEPOLARIZE2(0.125) 3 18 7 21 9 24 11 27 2 28 5 30
            DEPOLARIZE1(0.0125) 0 1 4 6 8 10 12 13 14 15 16 17 19 20 22 23 25 26 29 31 32 33 34 35
            TICK
            H 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35
            C_ZYX 0 1 2 3 4 5 6 7 8 9 10 11
            DEPOLARIZE1(0.0125) 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 0 1 2 3 4 5 6 7 8 9 10 11
            TICK
            X_ERROR(0.625) 15 19 22 25 29 33 12 16 17 20 23 26 31 32 13 14 18 21 24 27 28 30 34 35
            M 15 19 22 25 29 33
            DETECTOR(1.5, 2, 0) rec[-53] rec[-51] rec[-49] rec[-45] rec[-43] rec[-41] rec[-11] rec[-9] rec[-7] rec[-5] rec[-3] rec[-1]
            SHIFT_COORDS(0, 0, 1)
            M 12 16 17 20 23 26 31 32
            OBSERVABLE_INCLUDE(1) rec[-2]
            DETECTOR(1.5, 2, 0) rec[-19] rec[-17] rec[-15] rec[-5] rec[-3] rec[-1]
            DETECTOR(0.5, 5, 0) rec[-21] rec[-18] rec[-16] rec[-7] rec[-4] rec[-2]
            DETECTOR(0.5, -1, 0) rec[-22] rec[-20] rec[-8] rec[-6]
            SHIFT_COORDS(0, 0, 1)
            M 13 14 18 21 24 27 28 30 34 35
            OBSERVABLE_INCLUDE(1) rec[-4] rec[-2]
            DETECTOR(2.5, 3, 0) rec[-37] rec[-33] rec[-27] rec[-13] rec[-5] rec[-1]
            DETECTOR(0.5, 3, 0) rec[-41] rec[-39] rec[-35] rec[-29] rec[-26] rec[-15] rec[-12] rec[-9] rec[-7] rec[-3]
            SHIFT_COORDS(0, 0, 1)
            DEPOLARIZE1(0.0125) 0 1 2 3 4 5 6 7 8 9 10 11
            DEPOLARIZE1(0.25) 0 1 2 3 4 5 6 7 8 9 10 11
            TICK
        }
        R 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35
        X_ERROR(0.25) 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35
        DEPOLARIZE1(0.0125) 0 1 2 3 4 5 6 7 8 9 10 11
        DEPOLARIZE1(0.25) 0 1 2 3 4 5 6 7 8 9 10 11
        TICK
        H 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35
        C_ZYX 0 1 2 3 4 5 6 7 8 9 10 11
        DEPOLARIZE1(0.0125) 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 0 1 2 3 4 5 6 7 8 9 10 11
        TICK
        CZ 1 15 4 19 8 22 10 25 0 29 6 33
        DEPOLARIZE2(0.125) 1 15 4 19 8 22 10 25 0 29 6 33
        DEPOLARIZE1(0.0125) 2 3 5 7 9 11 12 13 14 16 17 18 20 21 23 24 26 27 28 30 31 32 34 35
        TICK
        CZ 2 15 5 19 7 22 9 25 3 29 11 33
        C_ZYX 0 1 4 6 8 10
        DEPOLARIZE2(0.125) 2 15 5 19 7 22 9 25 3 29 11 33
        DEPOLARIZE1(0.0125) 0 1 4 6 8 10 12 13 14 16 17 18 20 21 23 24 26 27 28 30 31 32 34 35
        TICK
        CZ 0 13 1 14 4 18 6 21 8 34 10 35
        C_ZYX 2 3 5 7 9 11
        DEPOLARIZE2(0.125) 0 13 1 14 4 18 6 21 8 34 10 35
        DEPOLARIZE1(0.0125) 2 3 5 7 9 11 12 15 16 17 19 20 22 23 24 25 26 27 28 29 30 31 32 33
        TICK
        CZ 3 18 7 21 9 24 11 27 2 28 5 30
        C_ZYX 0 1 4 6 8 10
        DEPOLARIZE2(0.125) 3 18 7 21 9 24 11 27 2 28 5 30
        DEPOLARIZE1(0.0125) 0 1 4 6 8 10 12 13 14 15 16 17 19 20 22 23 25 26 29 31 32 33 34 35
        TICK
        CZ 0 12 6 20 8 23 10 26 1 31 4 32
        C_ZYX 2 3 5 7 9 11
        DEPOLARIZE2(0.125) 0 12 6 20 8 23 10 26 1 31 4 32
        DEPOLARIZE1(0.0125) 2 3 5 7 9 11 13 14 15 16 17 18 19 21 22 24 25 27 28 29 30 33 34 35
        TICK
        CZ 2 16 3 17 5 20 11 26 7 31 9 32
        DEPOLARIZE2(0.125) 2 16 3 17 5 20 11 26 7 31 9 32
        DEPOLARIZE1(0.0125) 0 1 4 6 8 10 12 13 14 15 18 19 21 22 23 24 25 27 28 29 30 33 34 35
        TICK
        H 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35
        DEPOLARIZE1(0.0125) 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 0 1 2 3 4 5 6 7 8 9 10 11
        TICK
        X_ERROR(0.625) 15 19 22 25 29 33 13 14 18 21 24 27 28 30 34 35 12 16 17 20 23 26 31 32
        M 15 19 22 25 29 33
        SHIFT_COORDS(0, 0, 1)
        M 13 14 18 21 24 27 28 30 34 35
        OBSERVABLE_INCLUDE(1) rec[-4] rec[-2]
        DETECTOR(2.5, 1, 0) rec[-22] rec[-17] rec[-6] rec[-1]
        DETECTOR(1.5, 4, 0) rec[-23] rec[-21] rec[-18] rec[-7] rec[-5] rec[-2]
        DETECTOR(-0.5, 4, 0) rec[-25] rec[-20] rec[-9] rec[-4]
        DETECTOR(0.5, 1, 0) rec[-26] rec[-24] rec[-19] rec[-10] rec[-8] rec[-3]
        SHIFT_COORDS(0, 0, 1)
        M 12 16 17 20 23 26 31 32
        OBSERVABLE_INCLUDE(1) rec[-2]
        SHIFT_COORDS(0, 0, 1)
        DEPOLARIZE1(0.0125) 0 1 2 3 4 5 6 7 8 9 10 11
        DEPOLARIZE1(0.25) 0 1 2 3 4 5 6 7 8 9 10 11
        TICK
        R 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35
        X_ERROR(0.25) 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35
        DEPOLARIZE1(0.0125) 0 1 2 3 4 5 6 7 8 9 10 11
        DEPOLARIZE1(0.25) 0 1 2 3 4 5 6 7 8 9 10 11
        TICK
        H 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35
        C_ZYX 0 1 2 3 4 5 6 7 8 9 10 11
        DEPOLARIZE1(0.0125) 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 0 1 2 3 4 5 6 7 8 9 10 11
        TICK
        CZ 1 15 4 19 8 22 10 25 0 29 6 33
        DEPOLARIZE2(0.125) 1 15 4 19 8 22 10 25 0 29 6 33
        DEPOLARIZE1(0.0125) 2 3 5 7 9 11 12 13 14 16 17 18 20 21 23 24 26 27 28 30 31 32 34 35
        TICK
        CZ 2 15 5 19 7 22 9 25 3 29 11 33
        C_XYZ 0 1 4 6 8 10
        DEPOLARIZE2(0.125) 2 15 5 19 7 22 9 25 3 29 11 33
        DEPOLARIZE1(0.0125) 0 1 4 6 8 10 12 13 14 16 17 18 20 21 23 24 26 27 28 30 31 32 34 35
        TICK
        CZ 0 12 6 20 8 23 10 26 1 31 4 32
        C_XYZ 2 3 5 7 9 11
        DEPOLARIZE2(0.125) 0 12 6 20 8 23 10 26 1 31 4 32
        DEPOLARIZE1(0.0125) 2 3 5 7 9 11 13 14 15 16 17 18 19 21 22 24 25 27 28 29 30 33 34 35
        TICK
        CZ 2 16 3 17 5 20 11 26 7 31 9 32
        C_XYZ 0 1 4 6 8 10
        DEPOLARIZE2(0.125) 2 16 3 17 5 20 11 26 7 31 9 32
        DEPOLARIZE1(0.0125) 0 1 4 6 8 10 12 13 14 15 18 19 21 22 23 24 25 27 28 29 30 33 34 35
        TICK
        CZ 0 13 1 14 4 18 6 21 8 34 10 35
        C_XYZ 2 3 5 7 9 11
        DEPOLARIZE2(0.125) 0 13 1 14 4 18 6 21 8 34 10 35
        DEPOLARIZE1(0.0125) 2 3 5 7 9 11 12 15 16 17 19 20 22 23 24 25 26 27 28 29 30 31 32 33
        TICK
        CZ 3 18 7 21 9 24 11 27 2 28 5 30
        DEPOLARIZE2(0.125) 3 18 7 21 9 24 11 27 2 28 5 30
        DEPOLARIZE1(0.0125) 0 1 4 6 8 10 12 13 14 15 16 17 19 20 22 23 25 26 29 31 32 33 34 35
        TICK
        H 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35
        DEPOLARIZE1(0.0125) 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 0 1 2 3 4 5 6 7 8 9 10 11
        TICK
        X_ERROR(0.625) 15 19 22 25 29 33 12 16 17 20 23 26 31 32 13 14 18 21 24 27 28 30 34 35 0 1 2 3 4 5 6 7 8 9 10 11
        M 15 19 22 25 29 33
        DETECTOR(1.5, 2, 0) rec[-53] rec[-51] rec[-49] rec[-45] rec[-43] rec[-41] rec[-11] rec[-9] rec[-7] rec[-5] rec[-3] rec[-1]
        SHIFT_COORDS(0, 0, 1)
        M 12 16 17 20 23 26 31 32
        OBSERVABLE_INCLUDE(1) rec[-2]
        DETECTOR(1.5, 2, 0) rec[-19] rec[-17] rec[-15] rec[-5] rec[-3] rec[-1]
        DETECTOR(0.5, 5, 0) rec[-21] rec[-18] rec[-16] rec[-7] rec[-4] rec[-2]
        DETECTOR(0.5, -1, 0) rec[-22] rec[-20] rec[-8] rec[-6]
        SHIFT_COORDS(0, 0, 1)
        M 13 14 18 21 24 27 28 30 34 35
        OBSERVABLE_INCLUDE(1) rec[-4] rec[-2]
        DETECTOR(2.5, 3, 0) rec[-37] rec[-33] rec[-27] rec[-13] rec[-5] rec[-1]
        DETECTOR(0.5, 3, 0) rec[-41] rec[-39] rec[-35] rec[-29] rec[-26] rec[-15] rec[-12] rec[-9] rec[-7] rec[-3]
        SHIFT_COORDS(0, 0, 1)
        M 0 1 2 3 4 5 6 7 8 9 10 11
        DETECTOR(0, 0.5, 0) rec[-22] rec[-12]
        DETECTOR(0, 3.5, 0) rec[-21] rec[-11]
        DETECTOR(1, 0.5, 0) rec[-20] rec[-9] rec[-8]
        DETECTOR(1, 3.5, 0) rec[-19] rec[-6] rec[-5]
        DETECTOR(2, 0.5, 0) rec[-18] rec[-3]
        DETECTOR(2, 3.5, 0) rec[-17] rec[-1]
        DETECTOR(-0.5, 5, 0) rec[-16] rec[-10]
        DETECTOR(0.5, 2, 0) rec[-15] rec[-7]
        DETECTOR(1.5, 5, 0) rec[-14] rec[-4]
        DETECTOR(2.5, 2, 0) rec[-13] rec[-2]
        DETECTOR(1.5, 2, 0) rec[-35] rec[-33] rec[-31] rec[-27] rec[-25] rec[-23] rec[-8] rec[-7] rec[-6] rec[-3] rec[-2] rec[-1]
        OBSERVABLE_INCLUDE(1) rec[-11] rec[-10] rec[-5] rec[-4]
        TICK
    """)
