import collections
import dataclasses
from typing import List, DefaultDict, Any, Tuple, Dict, Union, Iterable, Set

import stim


@dataclasses.dataclass(frozen=True, unsafe_hash=True)
class Loc:
    """A spacetime location that an operation can occur at.

    Attributes:
        t: The time of the operation. Must be a whole number.
        p: The 2d location of the qubit targeted by the operation, represented as a complex number. The real and
            imaginary parts are the two coordinates, and do not have to be whole numbers.
    """
    t: int = 0
    p: complex = 0j

    def __lt__(self, other) -> bool:
        return (self.t, complex_key(self.p)) < (other.t, complex_key(other.p))

    def __add__(self, other: 'Loc') -> 'Loc':
        return Loc(t=self.t + other.t, p=self.p + other.p)


@dataclasses.dataclass(frozen=True, unsafe_hash=True)
class LocOp:
    """An operation with an associate spacetime location."""
    gate: str
    targets: Tuple[Loc, ...] = ()

    def __add__(self, offset: 'Loc') -> 'LocOp':
        return LocOp(gate=self.gate,
                     targets=tuple(t + offset for t in self.targets))


@dataclasses.dataclass
class CircuitCanvas:
    """A circuit represented as a dictionary from spacetime location to operation.

    Circuit canvases make it cheap to detect if an operation is at a location, but expensive to determine things like
    "what is the next operation after this one".
    """

    ops: Dict[Loc, LocOp] = dataclasses.field(default_factory=dict)

    def clear(self) -> None:
        """Resets the circuit contents to empty."""
        self.ops.clear()

    def min_time(self) -> int:
        """Determines the time of the earliest operation in the circuit."""
        return min((k.t for k in self.ops.keys()), default=-1)

    def max_time(self) -> int:
        """Determines the time of the latest operation in the circuit."""
        return max((k.t for k in self.ops.keys()), default=-1)

    @staticmethod
    def from_ops(ops: List[LocOp]) -> 'CircuitCanvas':
        """Creates a circuit canvas containing each spacetime operation from the given list."""
        result = CircuitCanvas()
        for op in ops:
            result.insert(op)
        return result
    
    def insert(self, op: LocOp) -> None:
        """Adds an operation to the canvas. There must not be an operation at the operation's location."""
        for loc in op.targets:
            assert loc not in self.ops
            self.ops[loc] = op

    def offset_by(self, offset: Loc) -> 'CircuitCanvas':
        """Returns a copy of the circuit canvas, but with all operation locations shifted by a given offset."""
        return CircuitCanvas(ops={
            k + offset: v + offset for k, v in self.ops.items()
        })

    def copy(self) -> 'CircuitCanvas':
        """Returns a copy of the circuit canvas that can be independently edited."""
        return CircuitCanvas(
            ops={k: v for k, v in self.ops.items()},
        )

    def __or__(self, other: Union['CircuitCanvas', LocOp]) -> 'CircuitCanvas':
        """Returns the conflict-free overlay of two circuit canvases.

        The union has all operations from both canvases.
        Wherever the canvases have overlapping operations, those operations must be identical.
        """
        return self.copy().__ior__(other)

    def __ior__(self, other: Union['CircuitCanvas', LocOp]) -> 'CircuitCanvas':
        """Overlays the given canvas over the receiving canvas, ensuring there are no conflicts.

        The result has all operations from both canvases.
        Wherever the canvases have overlapping operations, those operations must be identical.
        """
        if isinstance(other, LocOp):
            self.insert(other)
            return self
        elif isinstance(other, CircuitCanvas):
            for k, v in other.ops.items():
                assert k not in self.ops, f"{k!r} has {self.ops[k]!r} can't take {v!r}"
                self.ops[k] = v
            return self
        else:
            raise NotImplementedError()

    def used_coords_set(self) -> Set[complex]:
        """Returns the set of all locations targeted by any operation in the canvas."""
        return {k.p for k in self.ops.keys()}

    def to_stim_circuit(self, *, q2i: Dict[complex, int]) -> stim.Circuit:
        """Converts the canvas circuit into a stim circuit."""

        num_t = max((k.t for k in self.ops.keys()), default=-1) + 1
        
        circuit = stim.Circuit()

        moments: List[DefaultDict[str, List[int]]] = [collections.defaultdict(list) for _ in range(num_t)]

        for loc in sorted(self.ops.keys()):
            op_at = self.ops[loc]
            if op_at.targets[0] == loc:
                moments[loc.t][op_at.gate].extend(q2i[q.p] for q in op_at.targets)

        for t in range(num_t):
            if t:
                circuit.append("TICK")
            for gate in sorted(moments[t].keys()):
                circuit.append(gate, moments[t][gate])

        return circuit


def indexed_qubits_circuit_dict(positions: Iterable[complex]) -> Tuple[stim.Circuit, Dict[complex, int]]:
    """Assigns integer indices to the given positions.

    Args:
        positions: An iterable of unique complex numbers corresponding to qubit locations.

    Returns:
        A (circuit, dictionary) tuple.
        The circuit is a stim circuit containing QUBIT_COORDS instructions describing the qubits.
        The dictionary maps each complex value to its integer index.
    """
    q2i = {q: i for i, q in enumerate(sorted(positions, key=complex_key))}
    circuit = stim.Circuit()
    for q, i in q2i.items():
        circuit.append("QUBIT_COORDS", [i], [q.real, q.imag])
    return circuit, q2i


def complex_key(c: complex) -> Any:
    """Defines a sorting order for complex numbers."""
    return c.real != int(c.real), c.real, c.imag
