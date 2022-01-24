import stim

from hcb.tools.gen.circuit_canvas import indexed_qubits_circuit_dict
from hcb.tools.analysis.collecting import (
    DecodingProblem,
    DecodingProblemDesc,
)
from hcb.codes.surface.layout import (
    rect_surface_code_plan,
    rect_surface_code_measure_circuit,
    rect_surface_code_init_circuit,
    EPR_ANCILLA_LOC,
)
from hcb.codes.surface.stabilizer_plan_problem import StabilizerPlanProblem
from hcb.tools.gen.measurement_tracker import MeasurementTracker
from hcb.tools.gen.noise import NoiseModel


def generate_surface_code_memory_problem(*,
                                         distance: int,
                                         rounds: int,
                                         noise: float,
                                         basis: str,
                                         decoder: str) -> StabilizerPlanProblem:
    """Generates a simulated memory experiment.

    A memory experiment prepares a logical qubit and then preserves it through some number of rounds.

    Args:
        distance: The width and height of the patch of data qubits.
        rounds: The number of times to measure each measurement qubit.
        noise: The noise strength.
        basis: What logical state to prepare and measure. Can be:
            transversal_X: Fault tolerant preparation and measurement of |+>.
            transversal_Z: Fault tolerant preparation and measurement of |0>.
            frayed_Y: Unphysical noiseless preparation and measurement of |i>. There is still noise
                during the intermediate rounds.
            frayed_EPR_XX_ZZ: Unphysical noiseless preparation and measurement of |00> + |11>
                entangled with a noiseless ancilla. There is still noise during the intermediate
                rounds.
        decoder: The decoder to use for correcting errors.
    """
    assert basis in ["transversal_X", "transversal_Z", "frayed_Y", "frayed_EPR_XX_ZZ"]
    tracker = MeasurementTracker()
    layout_1 = rect_surface_code_plan(width=distance, height=distance)
    all_layouts = (layout_1,)
    body_rounds = rounds - (2 if "transversal" in basis else 0)
    all_data_coords = {
        q
        for plan in all_layouts
        for q in plan.data_coords_set()
    }
    all_used_coords = {
        q
        for plan in all_layouts
        for q in plan.used_coords_set()
    }
    if "EPR" in basis:
        all_used_coords.add(EPR_ANCILLA_LOC)

    # Generate circuit parts.
    head, q2i = indexed_qubits_circuit_dict(all_used_coords)
    head += rect_surface_code_init_circuit(layout=layout_1,
                                           tracker=tracker,
                                           q2i=q2i,
                                           basis=basis)
    body = stim.Circuit("TICK")
    body += layout_1.interpret_into_stim_circuit(out_tracker=tracker, q2i=q2i)
    body *= body_rounds
    tail = rect_surface_code_measure_circuit(layout=layout_1,
                                             tracker=tracker,
                                             q2i=q2i,
                                             basis=basis)

    # Add noise.
    if "frayed" not in basis:
        body = head + body + tail
        head.clear()
        tail.clear()
    ideal_circuit = head + body + tail
    noisy_qubits = {i for q, i in q2i.items() if q != EPR_ANCILLA_LOC}
    noisy_circuit = head + NoiseModel.SD6(noise).noisy_circuit(body, qs=noisy_qubits) + tail

    decoding_problem = DecodingProblem(
        circuit_maker=lambda: noisy_circuit,
        desc=DecodingProblemDesc(
            data_width=len({q.real for q in all_data_coords}),
            data_height=len({q.imag for q in all_data_coords}),
            code_distance=distance,
            num_qubits=len(all_used_coords),
            rounds=rounds,
            noise=noise,
            circuit_style=f"surface_code_memory",
            preserved_observable=basis,
            decoder=decoder,
        ),
    )

    return StabilizerPlanProblem(
        ideal_circuit=ideal_circuit,
        noisy_circuit=noisy_circuit,
        all_layouts=all_layouts,
        decoding_problem=decoding_problem,
    )
