import stim

from hcb.tools.gen.circuit_canvas import Loc, LocOp, CircuitCanvas


def test_canvas():
    c = CircuitCanvas()
    c.insert(LocOp('H', (Loc(5, 1j),)))
    assert c.to_stim_circuit(q2i={1j: 2}) == stim.Circuit("""
        TICK
        TICK
        TICK
        TICK
        TICK
        H 2
    """)
