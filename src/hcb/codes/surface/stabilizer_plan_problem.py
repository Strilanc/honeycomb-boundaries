import dataclasses
import pathlib
from typing import Tuple, Union

import stim

from hcb.tools.gen.viewer import stim_circuit_html_viewer
from hcb.tools.analysis.collecting import DecodingProblem
from hcb.tools.gen.stabilizer_plan import StabilizerPlan


def _print_wrote(path: pathlib.Path) -> None:
    text = str(path).replace('\\', '/')
    print(f"wrote file:///{text}")


@dataclasses.dataclass
class StabilizerPlanProblem:
    """A decoding problem with some additional debugging information for visualization."""
    ideal_circuit: stim.Circuit
    noisy_circuit: stim.Circuit
    all_layouts: Tuple[StabilizerPlan, ...]
    decoding_problem: DecodingProblem

    def sample_correct_count(self, shots: int) -> int:
        return self.decoding_problem.sample_correct_count(shots)

    def graphlike_code_distance(self) -> int:
        return len(self.noisy_circuit.shortest_graphlike_error(ignore_ungraphlike_errors=True))

    def write_debug_files(self, out_dir: Union[str, pathlib.Path]) -> None:
        out_dir = pathlib.Path(out_dir)
        path = out_dir / "stabilizer_configurations.svg"
        with open(path, "w") as f:
            print(StabilizerPlan.svg(*self.all_layouts), file=f)
        _print_wrote(path)

        path = out_dir / "circuit.stim"
        with open(path, "w") as f:
            print(self.ideal_circuit, file=f)
        _print_wrote(path)

        path = out_dir / "noisy_circuit.stim"
        with open(path, "w") as f:
            print(self.noisy_circuit, file=f)
        _print_wrote(path)

        path = out_dir / "circuit_layers.html"
        with open(path, 'w') as f:
            print(stim_circuit_html_viewer(self.noisy_circuit,
                                           width=500,
                                           height=800), file=f)
        _print_wrote(path)

        path = out_dir / "model.dem"
        det_model = self.noisy_circuit.detector_error_model(decompose_errors=True)
        with open(path, "w") as f:
            print(det_model, file=f)
        _print_wrote(path)

        kept = stim.Circuit()
        for instruction in self.ideal_circuit:
            if isinstance(instruction, stim.CircuitInstruction) and instruction.name == "MPP":
                continue
            kept.append(instruction)
        import stimcirq
        cirq_circuit = stimcirq.stim_circuit_to_cirq_circuit(kept)
        import cirq_web
        path = out_dir / "circuit_viewer.html"
        cirq_web.Circuit3D(cirq_circuit).generate_html_file(file_name=str(path))
        _print_wrote(path)
