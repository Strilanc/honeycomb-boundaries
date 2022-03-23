import dataclasses
from typing import Optional, Dict, Set, Tuple, Sequence, List
from enum import Enum

import stim

ANY_CLIFFORD_1_OPS = {"C_XYZ", "C_ZYX", "H", "H_YZ", "I"}
ANY_CLIFFORD_2_OPS = {"CX", "CY", "CZ", "XCX", "XCY", "XCZ", "YCX", "YCY", "YCZ"}
RESET_OPS = {"R", "RX", "RY"}
MEASURE_OPS = {"M", "MX", "MY"}
ANNOTATION_OPS = {"OBSERVABLE_INCLUDE", "DETECTOR", "SHIFT_COORDS", "QUBIT_COORDS", "TICK"}

STANDARD_GATE_SETS = ['SD6', 'SI1000']
EM3_LIKE_GATE_SETS = ['EM3_v1', 'EM3_v2', 'SDEM3', 'SIEM3000']

class MppErrorType(Enum):
    NONE = 0
    DEPOLARIZING = 1
    DEPHASING = 2
    CORRELATED = 3


@dataclasses.dataclass(frozen=True)
class NoiseModel:
    idle: float
    measure_reset_idle: float
    noisy_gates: Dict[str, float]
    any_clifford_1: Optional[float] = None
    any_clifford_2: Optional[float] = None
    mpp_error: MppErrorType = MppErrorType.DEPOLARIZING
    mpp_indep_flip_error: Optional[float] = None

    @staticmethod
    def dispatcher(noise_model_name: str, p: float):
        if noise_model_name == 'EM3_v2':
            return NoiseModel.EM3_v2(p)
        if noise_model_name == 'EM3_v1':
            return NoiseModel.EM3_v1(p)
        if noise_model_name == 'SDEM3':
            return NoiseModel.SDEM3(p)
        if noise_model_name == 'SIEM3000':
            return NoiseModel.SIEM3000(p)
        if noise_model_name == 'SD6':
            return NoiseModel.SD6(p)
        if noise_model_name == 'SI1000':
            return NoiseModel.SI1000(p)
        raise NotImplementedError(f'{noise_model_name}')

    @staticmethod
    def SD6(p: float) -> 'NoiseModel':
        return NoiseModel(
            any_clifford_1=p,
            idle=p,
            measure_reset_idle=0,
            noisy_gates={
                "CX": p,
                "R": p,
                "M": p,
            },
        )

    @staticmethod
    def PC3(p: float) -> 'NoiseModel':
        return NoiseModel(
            any_clifford_1=p,
            any_clifford_2=p,
            idle=p,
            measure_reset_idle=0,
            noisy_gates={
                "R": p,
                "M": p,
            },
        )

    @staticmethod
    def SI1000(p: float) -> 'NoiseModel':
        return NoiseModel(
            any_clifford_1=p / 10,
            idle=p / 10,
            measure_reset_idle=2 * p,
            noisy_gates={
                "CZ": p,
                "R": 2 * p,
                "M": 5 * p,
            },
        )

    @staticmethod
    def EM3_v1(p: float) -> 'NoiseModel':
        """EM3 with measurement flip errors independent of measurement target depolarization error."""
        return NoiseModel(
            idle=p,
            measure_reset_idle=0,
            any_clifford_1=p,
            noisy_gates={
                "R": p,
                "M": p,
                "MPP": p,
            },
        )

    @staticmethod
    def EM3_v2(p: float) -> 'NoiseModel':
        """EM3 with measurement flip errors correlated with measurement target depolarization error."""
        return NoiseModel(
            any_clifford_1=0,
            any_clifford_2=0,
            idle=p,
            measure_reset_idle=0,
            mpp_error=MppErrorType.CORRELATED,
            noisy_gates={
                "R": p/2,
                "M": p/2,
                "MPP": p,
            },
        )

    @staticmethod
    def SDEM3(p: float) -> 'NoiseModel':
        """EM3 with uncorrelated measurement flip errors and 2Q depolarizing errors

        ~Extremely~ similar to EM3_v1, but noisy gates to exactly match
        EM3_v3 and SIEM3000 for slightly more direct comparisons"""
        return NoiseModel(
            any_clifford_1=0,
            any_clifford_2=0,
            idle=p,
            measure_reset_idle=0,
            mpp_error=MppErrorType.DEPOLARIZING,
            mpp_indep_flip_error=0,
            noisy_gates={
                "R": p/2,
                "M": p/2,
                "MPP": p,
            },
        )

    @staticmethod
    def SIEM3000(p: float) -> 'NoiseModel':
        """Superconducting inspired entangling measurements

        Like EM3, but with correlated dephasing and independent bitflips
        on MPP operations, rather than depolarizing"""
        return NoiseModel(
            any_clifford_1=0,
            any_clifford_2=0,
            idle=p,
            measure_reset_idle=0,
            mpp_error=MppErrorType.DEPHASING,
            mpp_indep_flip_error=p,
            noisy_gates={
                "R": p/2,
                "M": p/2,
                "MPP": p,
            },
        )

    def noisy_op(self, op: stim.CircuitInstruction, p: float, ancilla: int) -> Tuple[stim.Circuit, stim.Circuit, stim.Circuit]:
        pre = stim.Circuit()
        mid = stim.Circuit()
        post = stim.Circuit()
        targets = op.targets_copy()
        args = op.gate_args_copy()
        if p > 0:
            if op.name in ANY_CLIFFORD_1_OPS:
                post.append("DEPOLARIZE1", targets, p)
            elif op.name in ANY_CLIFFORD_2_OPS:
                post.append("DEPOLARIZE2", targets, p)
            elif op.name in RESET_OPS or op.name in MEASURE_OPS:
                if op.name in RESET_OPS:
                    post.append("Z_ERROR" if op.name.endswith("X") else "X_ERROR", targets, p)
                if op.name in MEASURE_OPS:
                    pre.append("Z_ERROR" if op.name.endswith("X") else "X_ERROR", targets, p)
            elif op.name == "MPP":
                groups = group_mpp_targets(targets)
                assert all(len(g) in [1, 2] for g in groups)
                assert args == [] or args == [0]

                if self.mpp_error == MppErrorType.CORRELATED:
                    if self.mpp_indep_flip_error is not None:
                        raise ValueError("MPP independent flip errors aren't supported "
                                         "with correlated MPP errors")
                    for g in groups:
                        if len(g) == 2:
                            a, b = g
                            mid += parity_measurement_with_correlated_measurement_noise(
                                t1=a,
                                t2=b,
                                ancilla=ancilla,
                                mix_probability=p)
                        else:
                            assert len(g) == 1
                            if g[0].is_x_target:
                                pre.append("Z_ERROR", g[0].value, p)
                            else:
                                pre.append("X_ERROR", g[0].value, p)
                            mid.append("MPP", g, p)
                    return pre, mid, post
                else:
                    first_target = groups[0][0]

                    singlets = [t.value for g in groups if len(g) == 1 for t in g]
                    pairs = [t.value for g in groups if len(g) == 2 for t in g]
                    if singlets:
                        if self.mpp_error == MppErrorType.NONE:
                            pass
                        elif self.mpp_error == MppErrorType.DEPOLARIZING:
                            pre.append("DEPOLARIZE1", singlets, p)
                        elif self.mpp_error == MppErrorType.DEPHASING:
                            pz = p if first_target.is_z_target else 0
                            py = p if first_target.is_y_target else 0
                            px = p if first_target.is_x_target else 0
                            pauli_1_errors = px, py, pz
                            pre.append("PAULI_CHANNEL_1", singlets, pauli_1_errors)
                        else:
                            raise NotImplementedError(f"Mpp Error Type:"
                                                      f" {self.mpp_error}")
                    if pairs:
                        if self.mpp_error == MppErrorType.NONE:
                            # non-zero to fix the decomposition
                            pre.append("DEPOLARIZE2", pairs, 1E-15)
                        elif self.mpp_error == MppErrorType.DEPOLARIZING:
                            pre.append("DEPOLARIZE2", pairs, p)
                        elif self.mpp_error == MppErrorType.DEPHASING:
                            pzz = p if first_target.is_z_target else 0
                            pyy = p if first_target.is_y_target else 0
                            pxx = p if first_target.is_x_target else 0
                            # make single qubit cases non-zero for decomposition
                            pxi = pix = pyi = piy = piz = pzi = 1E-15
                            # make other cases = 0
                            pxy = pyx = pzx = pxz = pzy = pyz = 0
                            pauli_2_errors = [pix, piy, piz,
                                              pxi, pxx, pxy, pxz,
                                              pyi, pyx, pyy, pyz,
                                              pzi, pzx, pzy, pzz]
                            pre.append("PAULI_CHANNEL_2", pairs, pauli_2_errors)
                        else:
                            raise NotImplementedError(f"Mpp Error Type:"
                                                      f" {self.mpp_error}")
                    if self.mpp_indep_flip_error:
                        # flip error is orthogonal to the measurement axis
                        px = self.mpp_indep_flip_error if first_target.is_z_target else 0
                        pz = self.mpp_indep_flip_error if first_target.is_y_target else 0
                        py = self.mpp_indep_flip_error if first_target.is_x_target else 0
                        pauli_1_errors = px, py, pz
                        pre.append("PAULI_CHANNEL_1", singlets + pairs, pauli_1_errors)
                    args = [p]

            else:
                raise NotImplementedError(repr(op))
        mid.append(op.name, targets, args)
        return pre, mid, post

    def noisy_circuit(self,
                      circuit: stim.Circuit,
                      *,
                      qs: Optional[Set[int]] = None,
                      ) -> stim.Circuit:
        result = stim.Circuit()
        ancilla = circuit.num_qubits + 10

        current_moment_pre = stim.Circuit()
        current_moment_mid = stim.Circuit()
        current_moment_post = stim.Circuit()
        used_qubits: Set[int] = set()
        measured_or_reset_qubits: Set[int] = set()
        if qs is None:
            qs = set(range(circuit.num_qubits))

        def flush():
            nonlocal result
            if not current_moment_mid:
                return

            # Apply idle depolarization rules.
            idle_qubits = sorted(qs - used_qubits)
            if used_qubits and idle_qubits and self.idle > 0:
                current_moment_post.append("DEPOLARIZE1", idle_qubits, self.idle)
            idle_qubits = sorted(qs - measured_or_reset_qubits)
            if measured_or_reset_qubits and idle_qubits and self.measure_reset_idle > 0:
                current_moment_post.append("DEPOLARIZE1", idle_qubits, self.measure_reset_idle)

            # Move current noisy moment into result.
            result += current_moment_pre
            result += current_moment_mid
            result += current_moment_post
            used_qubits.clear()
            current_moment_pre.clear()
            current_moment_mid.clear()
            current_moment_post.clear()
            measured_or_reset_qubits.clear()

        for op in circuit:
            if isinstance(op, stim.CircuitRepeatBlock):
                flush()
                result += self.noisy_circuit(op.body_copy(), qs=qs) * op.repeat_count
            elif isinstance(op, stim.CircuitInstruction):
                if op.name == "TICK":
                    flush()
                    result.append("TICK", [])
                    continue

                if op.name in self.noisy_gates:
                    p = self.noisy_gates[op.name]
                elif self.any_clifford_1 is not None and op.name in ANY_CLIFFORD_1_OPS:
                    p = self.any_clifford_1
                elif self.any_clifford_2 is not None and op.name in ANY_CLIFFORD_2_OPS:
                    p = self.any_clifford_2
                elif op.name in ANNOTATION_OPS:
                    p = 0
                else:
                    raise NotImplementedError(repr(op))
                pre, mid, post = self.noisy_op(op, p, ancilla)
                current_moment_pre += pre
                current_moment_mid += mid
                current_moment_post += post

                # Ensure the circuit is not touching qubits multiple times per tick.
                touched_qubits = {
                    t.value
                    for t in op.targets_copy()
                    if t.is_x_target or t.is_y_target or t.is_z_target or t.is_qubit_target
                }
                if op.name in ANNOTATION_OPS:
                    touched_qubits.clear()
                assert touched_qubits.isdisjoint(used_qubits), "OVERLAPPING OPERATIONS IN:\n" + repr(current_moment_pre + current_moment_mid + current_moment_post)
                used_qubits |= touched_qubits
                if op.name in MEASURE_OPS or op.name in RESET_OPS:
                    measured_or_reset_qubits |= touched_qubits
            else:
                raise NotImplementedError(repr(op))
        flush()

        return result


def mix_probability_to_independent_component_probability(mix_probability: float, n: float) -> float:
    """Converts the probability of applying a full mixing channel to independent component probabilities.

    If each component is applied independently with the returned component probability, the overall effect
    is identical to, with probability `mix_probability`, uniformly picking one of the components to apply.

    Not that, unlike in other places in the code, the all-identity case is one of the components that can
    be picked when applying the error case.
    """
    return 0.5 - 0.5 * (1 - mix_probability) ** (1 / 2 ** (n - 1))


def parity_measurement_with_correlated_measurement_noise(
        *,
        t1: stim.GateTarget,
        t2: stim.GateTarget,
        ancilla: int,
        mix_probability: float) -> stim.Circuit:
    """Performs a noisy parity measurement.

    With probability mix_probability, applies a random element from

        {I1,X1,Y1,Z1}*{I2,X2,Y2,Z2}*{no flip, flip}

    Note that, unlike in other places in the code, the all-identity term is one of the possible
    samples when the error occurs.
    """

    ind_p = mix_probability_to_independent_component_probability(mix_probability, 5)

    # Generate all possible combinations of (non-identity) channels.  Assumes triple of targets
    # with last element corresponding to measure qubit.
    circuit = stim.Circuit()
    circuit.append('R', [ancilla])
    if t1.is_x_target:
        circuit.append('XCX', [t1.value, ancilla])
    if t1.is_y_target:
        circuit.append('YCX', [t1.value, ancilla])
    if t1.is_z_target:
        circuit.append('ZCX', [t1.value, ancilla])
    if t2.is_x_target:
        circuit.append('XCX', [t2.value, ancilla])
    if t2.is_y_target:
        circuit.append('YCX', [t2.value, ancilla])
    if t2.is_z_target:
        circuit.append('ZCX', [t2.value, ancilla])

    first_targets = ["I", stim.target_x(t1.value), stim.target_y(t1.value), stim.target_z(t1.value)]
    second_targets = ["I", stim.target_x(t2.value), stim.target_y(t2.value), stim.target_z(t2.value)]
    measure_targets = ["I", stim.target_x(ancilla)]

    errors = []
    for first_target in first_targets:
        for second_target in second_targets:
            for measure_target in measure_targets:
                error = []
                if first_target != "I":
                    error.append(first_target)
                if second_target != "I":
                    error.append(second_target)
                if measure_target != "I":
                    error.append(measure_target)

                if len(error) > 0:
                    errors.append(error)

    for error in errors:
        circuit.append("CORRELATED_ERROR", error, ind_p)

    circuit.append('M', [ancilla])

    return circuit


def group_mpp_targets(targets: Sequence[stim.GateTarget]) -> List[List[stim.GateTarget]]:
    groups = []
    start = 0
    while start < len(targets):
        end = start + 1
        while end < len(targets) and targets[end].is_combiner:
            end += 2
        groups.append(list(targets[start:end:2]))
        start = end
    return groups
