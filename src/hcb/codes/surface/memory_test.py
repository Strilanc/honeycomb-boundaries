import stim

from hcb.codes import generate_surface_code_memory_problem
from hcb.tools.analysis.collecting import DecodingProblemDesc


def test_generate_surface_code_memory_problem_transversal_Z():
    problem = generate_surface_code_memory_problem(
        distance=3,
        rounds=12,
        noise=0.001,
        basis="transversal_Z",
        decoder='pymatching',
    )

    assert len(problem.all_layouts) == 1
    assert problem.decoding_problem.desc == DecodingProblemDesc(
        data_width=3,
        data_height=3,
        code_distance=3,
        num_qubits=17,
        rounds=12,
        noise=0.001,
        circuit_style='surface_code_memory',
        preserved_observable='transversal_Z',
        decoder='pymatching',
    )
    assert problem.graphlike_code_distance() == 3
    assert 900 < problem.sample_correct_count(1000) < 1000


def test_generate_surface_code_memory_problem_transversal_X():
    problem = generate_surface_code_memory_problem(
        distance=3,
        rounds=12,
        noise=0.001,
        basis="transversal_X",
        decoder='pymatching',
    )

    assert len(problem.all_layouts) == 1
    assert problem.decoding_problem.desc == DecodingProblemDesc(
        data_width=3,
        data_height=3,
        code_distance=3,
        num_qubits=17,
        rounds=12,
        noise=0.001,
        circuit_style='surface_code_memory',
        preserved_observable='transversal_X',
        decoder='pymatching',
    )
    assert 900 < problem.sample_correct_count(1000) < 1000


def test_generate_surface_code_memory_problem_frayed_EPR_XX_ZZ():
    problem = generate_surface_code_memory_problem(
        distance=3,
        rounds=12,
        noise=0.001,
        basis="frayed_EPR_XX_ZZ",
        decoder='pymatching',
    )

    assert len(problem.all_layouts) == 1
    assert len(problem.noisy_circuit.detector_error_model(decompose_errors=True).shortest_graphlike_error()) == 3
    assert problem.decoding_problem.desc == DecodingProblemDesc(
        data_width=3,
        data_height=3,
        code_distance=3,
        num_qubits=18,
        rounds=12,
        noise=0.001,
        circuit_style='surface_code_memory',
        preserved_observable='frayed_EPR_XX_ZZ',
        decoder='pymatching',
    )
    assert 900 < problem.sample_correct_count(1000) < 1000


def test_generate_surface_code_memory_problem_frayed_Y():
    problem = generate_surface_code_memory_problem(
        distance=3,
        rounds=12,
        noise=0.001,
        basis="frayed_Y",
        decoder='pymatching',
    )

    assert len(problem.all_layouts) == 1
    assert problem.decoding_problem.desc == DecodingProblemDesc(
        data_width=3,
        data_height=3,
        code_distance=3,
        num_qubits=17,
        rounds=12,
        noise=0.001,
        circuit_style='surface_code_memory',
        preserved_observable='frayed_Y',
        decoder='pymatching',
    )
    assert 900 < problem.sample_correct_count(1000) < 1000


def test_generate_memory_exact_circuit():
    assert generate_surface_code_memory_problem(
        distance=3,
        rounds=12,
        noise=0.001,
        basis="transversal_Z",
        decoder='pymatching',
    ).noisy_circuit == stim.Circuit("""
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(0, 1) 1
        QUBIT_COORDS(0, 2) 2
        QUBIT_COORDS(1, 0) 3
        QUBIT_COORDS(1, 1) 4
        QUBIT_COORDS(1, 2) 5
        QUBIT_COORDS(2, 0) 6
        QUBIT_COORDS(2, 1) 7
        QUBIT_COORDS(2, 2) 8
        QUBIT_COORDS(-0.5, 1.5) 9
        QUBIT_COORDS(0.5, -0.5) 10
        QUBIT_COORDS(0.5, 0.5) 11
        QUBIT_COORDS(0.5, 1.5) 12
        QUBIT_COORDS(1.5, 0.5) 13
        QUBIT_COORDS(1.5, 1.5) 14
        QUBIT_COORDS(1.5, 2.5) 15
        QUBIT_COORDS(2.5, 0.5) 16
        R 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
        X_ERROR(0.001) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
        TICK
        H 10 12 13 15
        DEPOLARIZE1(0.001) 10 12 13 15 0 1 2 3 4 5 6 7 8 9 11 14 16
        TICK
        CX 0 11 4 14 6 16 12 1 13 3 15 5
        DEPOLARIZE2(0.001) 0 11 4 14 6 16 12 1 13 3 15 5
        DEPOLARIZE1(0.001) 2 7 8 9 10
        TICK
        CX 1 11 5 14 7 16 12 4 13 6 15 8
        DEPOLARIZE2(0.001) 1 11 5 14 7 16 12 4 13 6 15 8
        DEPOLARIZE1(0.001) 0 2 3 9 10
        TICK
        CX 1 9 3 11 7 14 10 0 12 2 13 4
        DEPOLARIZE2(0.001) 1 9 3 11 7 14 10 0 12 2 13 4
        DEPOLARIZE1(0.001) 5 6 8 15 16
        TICK
        CX 2 9 4 11 8 14 10 3 12 5 13 7
        DEPOLARIZE2(0.001) 2 9 4 11 8 14 10 3 12 5 13 7
        DEPOLARIZE1(0.001) 0 1 6 15 16
        TICK
        H 10 12 13 15
        DEPOLARIZE1(0.001) 10 12 13 15 0 1 2 3 4 5 6 7 8 9 11 14 16
        TICK
        X_ERROR(0.001) 9 10 11 12 13 14 15 16
        M 9 10 11 12 13 14 15 16
        DETECTOR(-0.5, 1.5, 0) rec[-8]
        DETECTOR(0.5, 0.5, 0) rec[-6]
        DETECTOR(1.5, 1.5, 0) rec[-3]
        DETECTOR(2.5, 0.5, 0) rec[-1]
        SHIFT_COORDS(0, 0, 1)
        DEPOLARIZE1(0.001) 0 1 2 3 4 5 6 7 8
        REPEAT 10 {
            TICK
            R 9 10 11 12 13 14 15 16
            X_ERROR(0.001) 9 10 11 12 13 14 15 16
            DEPOLARIZE1(0.001) 0 1 2 3 4 5 6 7 8
            TICK
            H 10 12 13 15
            DEPOLARIZE1(0.001) 10 12 13 15 0 1 2 3 4 5 6 7 8 9 11 14 16
            TICK
            CX 0 11 4 14 6 16 12 1 13 3 15 5
            DEPOLARIZE2(0.001) 0 11 4 14 6 16 12 1 13 3 15 5
            DEPOLARIZE1(0.001) 2 7 8 9 10
            TICK
            CX 1 11 5 14 7 16 12 4 13 6 15 8
            DEPOLARIZE2(0.001) 1 11 5 14 7 16 12 4 13 6 15 8
            DEPOLARIZE1(0.001) 0 2 3 9 10
            TICK
            CX 1 9 3 11 7 14 10 0 12 2 13 4
            DEPOLARIZE2(0.001) 1 9 3 11 7 14 10 0 12 2 13 4
            DEPOLARIZE1(0.001) 5 6 8 15 16
            TICK
            CX 2 9 4 11 8 14 10 3 12 5 13 7
            DEPOLARIZE2(0.001) 2 9 4 11 8 14 10 3 12 5 13 7
            DEPOLARIZE1(0.001) 0 1 6 15 16
            TICK
            H 10 12 13 15
            DEPOLARIZE1(0.001) 10 12 13 15 0 1 2 3 4 5 6 7 8 9 11 14 16
            TICK
            X_ERROR(0.001) 9 10 11 12 13 14 15 16
            M 9 10 11 12 13 14 15 16
            DETECTOR(-0.5, 1.5, 0) rec[-16] rec[-8]
            DETECTOR(0.5, -0.5, 0) rec[-15] rec[-7]
            DETECTOR(0.5, 0.5, 0) rec[-14] rec[-6]
            DETECTOR(0.5, 1.5, 0) rec[-13] rec[-5]
            DETECTOR(1.5, 0.5, 0) rec[-12] rec[-4]
            DETECTOR(1.5, 1.5, 0) rec[-11] rec[-3]
            DETECTOR(1.5, 2.5, 0) rec[-10] rec[-2]
            DETECTOR(2.5, 0.5, 0) rec[-9] rec[-1]
            SHIFT_COORDS(0, 0, 1)
            DEPOLARIZE1(0.001) 0 1 2 3 4 5 6 7 8
        }
        R 9 10 11 12 13 14 15 16
        X_ERROR(0.001) 9 10 11 12 13 14 15 16
        DEPOLARIZE1(0.001) 0 1 2 3 4 5 6 7 8
        TICK
        H 10 12 13 15
        DEPOLARIZE1(0.001) 10 12 13 15 0 1 2 3 4 5 6 7 8 9 11 14 16
        TICK
        CX 0 11 4 14 6 16 12 1 13 3 15 5
        DEPOLARIZE2(0.001) 0 11 4 14 6 16 12 1 13 3 15 5
        DEPOLARIZE1(0.001) 2 7 8 9 10
        TICK
        CX 1 11 5 14 7 16 12 4 13 6 15 8
        DEPOLARIZE2(0.001) 1 11 5 14 7 16 12 4 13 6 15 8
        DEPOLARIZE1(0.001) 0 2 3 9 10
        TICK
        CX 1 9 3 11 7 14 10 0 12 2 13 4
        DEPOLARIZE2(0.001) 1 9 3 11 7 14 10 0 12 2 13 4
        DEPOLARIZE1(0.001) 5 6 8 15 16
        TICK
        CX 2 9 4 11 8 14 10 3 12 5 13 7
        DEPOLARIZE2(0.001) 2 9 4 11 8 14 10 3 12 5 13 7
        DEPOLARIZE1(0.001) 0 1 6 15 16
        TICK
        H 10 12 13 15
        DEPOLARIZE1(0.001) 10 12 13 15 0 1 2 3 4 5 6 7 8 9 11 14 16
        TICK
        X_ERROR(0.001) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
        M 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
        DETECTOR(-0.5, 1.5, 0) rec[-25] rec[-8]
        DETECTOR(0.5, -0.5, 0) rec[-24] rec[-7]
        DETECTOR(0.5, 0.5, 0) rec[-23] rec[-6]
        DETECTOR(0.5, 1.5, 0) rec[-22] rec[-5]
        DETECTOR(1.5, 0.5, 0) rec[-21] rec[-4]
        DETECTOR(1.5, 1.5, 0) rec[-20] rec[-3]
        DETECTOR(1.5, 2.5, 0) rec[-19] rec[-2]
        DETECTOR(2.5, 0.5, 0) rec[-18] rec[-1]
        SHIFT_COORDS(0, 0, 1)
        DETECTOR(-0.5, 1.5, 0) rec[-16] rec[-15] rec[-8]
        DETECTOR(0.5, 0.5, 0) rec[-17] rec[-16] rec[-14] rec[-13] rec[-6]
        DETECTOR(1.5, 1.5, 0) rec[-13] rec[-12] rec[-10] rec[-9] rec[-3]
        DETECTOR(2.5, 0.5, 0) rec[-11] rec[-10] rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-17] rec[-14] rec[-11]
    """)
