import dataclasses
import math
from typing import List, Tuple, Dict, Iterable, Optional, Set, Any, AbstractSet

import stim

from hcb.tools.gen.circuit_canvas import CircuitCanvas, Loc, LocOp, complex_key
from hcb.tools.gen.measurement_tracker import MeasurementTracker, Prev
from hcb.tools.gen.viewer import tag_str

ALL_MEASURE_OPS = {"M", "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ"}


def append_mpp(*,
               basis: str = "",
               targets: Iterable[complex] = (),
               q2i: Dict[complex, int],
               out_circuit: stim.Circuit,
               xs: Iterable[complex] = (),
               ys: Iterable[complex] = (),
               zs: Iterable[complex] = (),
               noise: float = 0) -> None:
    targets = tuple(targets)
    if set(xs) & set(zs):
        common = set(xs) & set(zs)
        ys = tuple(ys) + tuple(x for x in xs if x in common)
        xs = [x for x in xs if x not in common]
        zs = [z for z in zs if z not in common]
    if targets:
        assert basis in "XYZ"
        if basis == "X":
            xs += targets
        if basis == "Y":
            ys += targets
        if basis == "Z":
            zs += targets

    int_targets = []
    for q in xs:
        int_targets.append(stim.target_x(q2i[q]))
        int_targets.append(stim.target_combiner())
    for q in ys:
        int_targets.append(stim.target_y(q2i[q]))
        int_targets.append(stim.target_combiner())
    for q in zs:
        int_targets.append(stim.target_z(q2i[q]))
        int_targets.append(stim.target_combiner())
    int_targets.pop()

    out_circuit.append("MPP", int_targets, noise if noise else [])


@dataclasses.dataclass(frozen=True, unsafe_hash=True)
class StabilizerPlanElement:
    """A stabilizer and information on how to go about measuring it.

    A stabilizer configuration is just a collection of these elements.

    Attributes:
        basis: The basis of the stabilizer to measure. Must be "X" or "Z".
        measurement_qubit: The ancillary qubit the data qubits are being xored into.
        data_qubit_order: The data qubits to xor into the measurement qubit, in the order they should xored into it.
            The list may contain None values, which are not a data qubit but rather an indication that there should be
            a one moment delay until the next data qubit is xored into the measurement qubit.
    """

    bases: Tuple[str, ...]
    measurement_qubit: complex
    data_qubit_order: Tuple[Optional[complex], ...]
    measurement_qubit_is_artificial: bool = False

    def __post_init__(self):
        assert isinstance(self.bases, tuple)
        assert len(self.bases) == len(self.data_qubit_order)

    def with_clipped_data_qubits(self, clipped_data_qubits: AbstractSet[complex]) -> 'StabilizerPlanElement':
        return StabilizerPlanElement(
            bases=self.bases,
            measurement_qubit=self.measurement_qubit,
            data_qubit_order=tuple(None if q in clipped_data_qubits else q for q in self.data_qubit_order),
        )

    def offset_by(self, offset: complex) -> 'StabilizerPlanElement':
        return StabilizerPlanElement(
            bases=self.bases,
            measurement_qubit=self.measurement_qubit + offset,
            data_qubit_order=tuple(e + offset for e in self.data_qubit_order),
        )

    def xyz_data_coords(self) -> Tuple[complex, ...]:
        return tuple(
            q
            for b, q in zip(self.bases, self.data_qubit_order)
            if b in 'XYZ'
        )

    def append_mpp(self, *, out_circuit: stim.Circuit, q2i: Dict[complex, int], noise: float = 0):
        append_mpp(out_circuit=out_circuit,
                   xs=[q for q, b in zip(self.data_qubit_order, self.bases) if q is not None and b == "X"],
                   ys=[q for q, b in zip(self.data_qubit_order, self.bases) if q is not None and b == "Y"],
                   zs=[q for q, b in zip(self.data_qubit_order, self.bases) if q is not None and b == "Z"],
                   q2i=q2i,
                   noise=noise)

    def is_leaf(self) -> bool:
        return sum(bool(e is not None) for e in self.data_qubit_order) == 1

    def data_coords_set(self) -> Set[complex]:
        return {e for e in self.data_qubit_order if e is not None}

    def used_coords_set(self) -> Set[complex]:
        if self.measurement_qubit_is_artificial:
            return self.data_coords_set()
        return self.data_coords_set() | {self.measurement_qubit}

    def common_basis(self) -> Optional[str]:
        s = {b for q, b in zip(self.data_qubit_order, self.bases) if q is not None}
        if len(s) == 1:
            b, = s
            return b
        return None

    def operations_to_canvas(self) -> CircuitCanvas:
        """Returns a circuit canvas containing a circuit that measures this stabilizer."""
        c = self.common_basis()
        rev = -1 if c == "X" else +1
        result = CircuitCanvas()
        mq = self.measurement_qubit

        t = 0
        result.insert(LocOp(gate="R", targets=(Loc(t=t, p=mq),)))
        t += 1
        if c == "X":
            result.insert(LocOp(gate="H", targets=(Loc(t=t, p=mq),)))
        t += 1
        for k, data_qubit in enumerate(self.data_qubit_order):
            if data_qubit is not None:
                if c:
                    result.insert(LocOp(gate="CX", targets=(Loc(t=t, p=data_qubit), Loc(t=t, p=mq))[::rev]))
                else:
                    result.insert(LocOp(gate=f"{self.bases[k]}CX", targets=(Loc(t=t, p=data_qubit), Loc(t=t, p=mq))[::rev]))
            t += 1
        if c == "X":
            result.insert(LocOp(gate="H", targets=(Loc(t=t, p=mq),)))
        t += 1
        result.insert(LocOp(gate="M", targets=(Loc(t=t, p=mq),)))
        return result


@dataclasses.dataclass(frozen=True)
class StabilizerPlan:
    """A configuration of stabilizers to measure simultaneously.

    The plan corresponds to a circuit where all the stabilizer elements' circuits are overlayed.
    It is not verified that the element circuits don't interfere with each other (eg. bad interleaving of the operations
    to measure their respective stabilizers).
    """
    elements: Tuple[StabilizerPlanElement, ...]
    observables: Tuple[StabilizerPlanElement, ...] = ()

    def used_coords_set(self) -> Set[complex]:
        result = set()
        for e in self.elements:
            result |= e.used_coords_set()
        for e in self.observables:
            result |= e.used_coords_set()
        return result

    def do_mpp_measurements(self,
                            *,
                            out_circuit: stim.Circuit = None,
                            initial_detect_vs_exceptions: Optional[Dict[Any, Optional[List[Any]]]] = None,
                            reset_xs: Iterable[complex] = (),
                            reset_ys: Iterable[complex] = (),
                            reset_zs: Iterable[complex] = (),
                            q2i: Dict[complex, int],
                            tracker: MeasurementTracker,
                            noise: float = 0,
                            detect_vs_prev: bool = False,
                            tick: bool = False,
                            reps: int = 1) -> stim.Circuit:
        assert reps >= 0 and isinstance(reps, int)
        if initial_detect_vs_exceptions is None:
            initial_detect_vs_exceptions = {}

        body = stim.Circuit()
        reset_xs = tuple(sorted(reset_xs, key=complex_key))
        reset_ys = tuple(sorted(reset_ys, key=complex_key))
        reset_zs = tuple(sorted(reset_zs, key=complex_key))
        if reset_xs or reset_ys or reset_zs:
            body.append("R", [q2i[q] for q in reset_xs + reset_ys + reset_zs])
            if noise:
                body.append("DEPOLARIZE1", sorted(q2i.values()), noise)
            if tick:
                body.append("TICK")
        if reset_xs or reset_ys:
            body.append("H", [q2i[q] for q in reset_xs])
            body.append("H_YZ", [q2i[q] for q in reset_ys])
            if noise:
                body.append("DEPOLARIZE1", sorted(q2i.values()), noise)
            if tick:
                body.append("TICK")

        if reps == 0:
            if out_circuit is not None:
                out_circuit += body
            return body

        for e in self.sorted_elements():
            e.append_mpp(out_circuit=body, q2i=q2i, noise=noise)
            tracker.add_measurements(e.measurement_qubit)
        if detect_vs_prev or initial_detect_vs_exceptions:
            for e in self.sorted_elements():
                m = e.measurement_qubit
                comparisons = initial_detect_vs_exceptions.get(m, [Prev(m)])
                if comparisons is None:
                    continue
                assert isinstance(comparisons, list), f"Vs exception must be a list but got {comparisons!r} for {m!r}"
                tracker.append_detector(m, *comparisons,
                                        out_circuit=body,
                                        coords=[m.real, m.imag, 0])
        if noise:
            body.append("DEPOLARIZE1", sorted(q2i.values()), noise)
        if tick:
            body.append("TICK")
            body.append("SHIFT_COORDS", [], [0, 0, 1])

        if reps > 1:
            if initial_detect_vs_exceptions:
                body += self.do_mpp_measurements(detect_vs_prev=detect_vs_prev,
                                                 tick=tick,
                                                 noise=noise,
                                                 q2i=q2i,
                                                 tracker=tracker) * (reps - 1)
            else:
                body *= reps

        if out_circuit is not None:
            out_circuit += body
        return body

    def data_coords_set(self) -> Set[complex]:
        result = set()
        for e in self.elements:
            for q in e.data_qubit_order:
                if q is not None:
                    result.add(q)
        return result

    def measure_coords_set(self) -> Set[complex]:
        return {e.measurement_qubit for e in self.elements if not e.measurement_qubit_is_artificial}

    def sorted_data_coords(self) -> List[complex]:
        return sorted(self.data_coords_set(), key=complex_key)

    def sorted_elements(self) -> List[StabilizerPlanElement]:
        return sorted(self.elements, key=lambda v: complex_key(v.measurement_qubit))

    def to_canvas(self) -> CircuitCanvas:
        canvas = CircuitCanvas()
        se = self.sorted_elements()
        for e in se:
            canvas |= e.operations_to_canvas()
        return canvas

    def interpret_into_stim_circuit(self,
                                    *,
                                    edited_canvas: Optional[CircuitCanvas] = None,
                                    out_circuit: Optional[stim.Circuit] = None,
                                    out_tracker: MeasurementTracker,
                                    q2i: Dict[complex, int],
                                    dont_add_detectors: bool = False) -> stim.Circuit:
        if out_circuit is None:
            out_circuit = stim.Circuit()
        if edited_canvas is None:
            edited_canvas = self.to_canvas()
        for loc in sorted(edited_canvas.ops.keys()):
            val = edited_canvas.ops[loc]
            if val.gate in ALL_MEASURE_OPS and val.targets[0] == loc:
                out_tracker.add_measurements(*[q.p for q in val.targets])
        out_circuit += edited_canvas.to_stim_circuit(q2i=q2i)
        if not dont_add_detectors:
            for e in self.sorted_elements():
                out_tracker.append_detector(e.measurement_qubit,
                                            Prev(e.measurement_qubit),
                                            out_circuit=out_circuit,
                                            coords=[e.measurement_qubit.real, e.measurement_qubit.imag, 0])
        out_circuit.append("SHIFT_COORDS", [], [0, 0, 1])
        return out_circuit

    def bounding_box(self) -> Tuple[complex, complex]:
        min_r = min((e.real for e in self.used_coords_set()), default=0)
        min_i = min((e.imag for e in self.used_coords_set()), default=0)
        max_r = max((e.real for e in self.used_coords_set()), default=0)
        max_i = max((e.imag for e in self.used_coords_set()), default=0)
        min_r -= 1
        max_r += 1
        min_i -= 1
        max_i += 1
        min_r = min(0, min_r)
        min_i = min(0, min_i)
        return min_r + min_i * 1j, max_r + max_i * 1j

    def svg(*plans: 'StabilizerPlan',
            canvas_height: int = 500,
            show_order: bool = True) -> str:
        """Returns a picture of the stabilizers measured by this plan.

        In the picture, black and red polygons correspond to X and Z stabilizers.
        """
        boxes = [plan.bounding_box() for plan in plans]
        box_width = max(((b - a).real for a, b in boxes), default=0)
        box_width *= 1.1
        height = max(((b - a).imag for a, b in boxes), default=0)
        width = box_width * len(plans)
        scale_factor = canvas_height / max(height, 1)
        canvas_width = int(math.ceil(canvas_height * (width / height)))

        def transform_pt(plan_i: int, pt: complex) -> complex:
            min_ri = boxes[plan_i][0]
            pt -= min_ri
            pt += box_width * plan_i
            pt *= scale_factor
            return pt

        def transform_dif(dif: complex) -> complex:
            return dif * scale_factor

        def pt(plan_i: int, q: complex) -> str:
            return f"{transform_pt(plan_i, q).real},{transform_pt(plan_i, q).imag}"

        def dt(q: complex) -> str:
            return f"{transform_dif(q).real},{transform_dif(q).imag}"

        lines = [f"""<svg viewBox="0 0 {canvas_width} {canvas_height}" xmlns="http://www.w3.org/2000/svg">"""]

        # Draw each plan element as a polygon.
        clip_path_id = 0
        BASE_COLORS = {"X": '#FFB0B0', "Z": '#B0B0FF', "Y": '#B0FFB0', None: "gray", '_': "black"}
        STRONG_BASE_COLORS = {"X": 'red', "Z": 'blue', "Y": 'green', None: "gray", '_': "black"}
        OBS_COLORS = ['#8000FF', '#FF8000']
        post_lines = []
        for plan_i, plan in enumerate(plans):
            for e in plan.elements:
                c = e.measurement_qubit
                dq = sorted(e.data_coords_set(), key=lambda p: math.atan2(p.imag - c.imag, p.real - c.real))
                common_basis = e.common_basis()
                fill_color = BASE_COLORS[common_basis]
                strong_fill_color = STRONG_BASE_COLORS[common_basis]

                # Reduce degenerate triangle to a semi circle.
                if len(dq) == 3:
                    for k in range(3):
                        if abs(dq[k-1] - (dq[k-2] + dq[k]) / 2) < 1e-4:
                            dq = [dq[k-2], dq[k]]
                            dq = sorted(dq, key=lambda p: math.atan2(p.imag - c.imag, p.real - c.real))
                            break


                if len(dq) == 1:
                    a, = dq
                    b = c
                    lines.append(f'<path '
                                 f'd="M{pt(plan_i, a)} '
                                 f'{pt(plan_i, b)}" '
                                 f'stroke-width="{abs(transform_dif(0.2))}" '
                                 f'stroke="{strong_fill_color}" '
                                 f'fill="none" />')
                elif len(dq) == 2:
                    a, b = dq
                    da = a - c
                    db = b - c
                    dab = math.atan2(da.imag, da.real) - math.atan2(db.imag, db.real)
                    dab %= math.pi * 2
                    if dab < math.pi:
                        a, b = b, a

                    c = (a + b) / 2
                    if abs(c - e.measurement_qubit) < 1e-4:
                        lines.append(f'<path '
                                     f'd="M{pt(plan_i, a)} '
                                     f'{pt(plan_i, b)}" '
                                     f'stroke-width="{abs(transform_dif(0.2))}" '
                                     f'stroke="{strong_fill_color}" '
                                     f'fill="none" />')
                    else:
                        lines.append(f'<path '
                                     f'd="M{pt(plan_i, a)} '
                                     f'a1,1 '
                                     f'0 0,0 '
                                     f'{dt(b - a)}" '
                                     f'fill="{fill_color}" '
                                     f'stroke-width="{abs(transform_dif(0.05))}" '
                                     f'stroke="black" />')
                else:
                    x = f'<path d="M{pt(plan_i, dq[-1])}'
                    for q in dq:
                        x += ' ' + pt(plan_i, q)
                    x += '"'
                    lines.append(f'{x} fill="{fill_color}" stroke="none" />')
                    post_lines.append(f'{x} stroke="black" stroke-width="{abs(transform_dif(0.05))}" fill="none" />')
                    if common_basis is None:
                        clip_path_id += 1
                        lines.append(f'<clipPath id="clipPath{clip_path_id}">')
                        lines.append(f'    {x} />')
                        lines.append(f'</clipPath>')
                        for k, q in enumerate(dq):
                            v = transform_pt(plan_i, q)
                            lines.append(f'<circle '
                                         f'clip-path="url(#clipPath{clip_path_id})" '
                                         f'cx="{v.real}" '
                                         f'cy="{v.imag}" '
                                         f'r="{abs(transform_dif(0.3))}" '
                                         f'fill="{BASE_COLORS[e.bases[k]]}" '
                                         f'stroke="none" />')
        lines += post_lines

        # Draw each element's measurement order as a zig zag arrow.
        if show_order:
            for plan_i, plan in enumerate(plans):
                for e in plan.elements:
                    c = e.measurement_qubit
                    if len(e.data_coords_set()) == 3:
                        c = 0
                        for q in e.data_coords_set():
                            c += q
                        c /= len(e.data_coords_set())
                    used = False

                    x = f'<path d="M'
                    arrow_color = "white"
                    delay = 0
                    prev = None
                    for q in e.data_qubit_order:
                        if q is not None:
                            v = q * 0.6 + c * 0.4
                            x += pt(plan_i, v) + ' '
                            v = transform_pt(plan_i, v)
                            if not used:
                                lines.append(f'<circle cx="{v.real}" cy="{v.imag}" r="2" fill="{arrow_color}" />')
                                used = True
                            for d in range(delay):
                                if prev is None:
                                    prev = v
                                v2 = (prev + v) / 2
                                lines.append(f'<circle cx="{v2.real}" cy="{v2.imag}" r="{d*2+4}" stroke="yellow" fill="none" />')
                            delay = 0
                            prev = v
                        else:
                            delay += 1
                    x = x.strip()
                    x += f'" fill="none" stroke="{arrow_color}" />'
                    lines.append(x)

        for plan_i, p in enumerate(plans):
            all_coords = {q for e in p.elements for q in e.data_coords_set()}
            for q in all_coords:
                xy = transform_pt(plan_i, q)
                lines.append(f'<circle '
                             f'cx="{xy.real}" '
                             f'cy="{xy.imag}" '
                             f'r="{abs(transform_dif(0.1))}" '
                             f'stroke-width="{abs(transform_dif(0.05))}" '
                             f'stroke="black" '
                             f'fill="black" />')
            all_coords = {e.measurement_qubit for e in p.elements if not e.measurement_qubit_is_artificial}
            for q in all_coords:
                xy = transform_pt(plan_i, q)
                lines.append(f'<circle '
                             f'cx="{xy.real}" '
                             f'cy="{xy.imag}" '
                             f'r="{abs(transform_dif(0.1))}" '
                             f'stroke="black" '
                             f'stroke-width="{abs(transform_dif(0.05))}" '
                             f'fill="white" />')

        for plan_i, p in enumerate(plans):
            for obs_i, obs in enumerate(p.observables):
                path_text = '<path d="M'
                for q, b in zip(obs.data_qubit_order, obs.bases):
                    path_text += pt(plan_i, q + (1 + 1j)*0.1*obs_i) + ' '
                path_text += f'" stroke-width="{abs(transform_dif(0.3))}" stroke="{OBS_COLORS[obs_i]}" opacity="0.9" fill="none" />'
                lines.append(path_text)
        for plan_i, p in enumerate(plans):
            for obs_i, obs in enumerate(p.observables):
                for q, b in zip(obs.data_qubit_order[1:-1], obs.bases[1:-1]):
                    c = transform_pt(plan_i, q + (1 + 1j)*0.1*obs_i)
                    r = abs(transform_dif(0.2 + 0j))
                    lines.append(tag_str(
                        "circle",
                        cx=c.real,
                        cy=c.imag,
                        r=r,
                        fill=STRONG_BASE_COLORS[b],
                        stroke="black"))

        # Draw coordinates.
        all_used_coords = {q for p in plans for e in p.elements for q in e.used_coords_set()}
        xs = {c.real for c in all_used_coords}
        ys = {c.imag for c in all_used_coords}
        for plan_i, p in enumerate(plans):
            for x in xs:
                if x != int(x):
                    continue
                c = transform_pt(plan_i, x + boxes[plan_i][0].imag*1j)
                lines.append(tag_str(
                    "text",
                     x=c.real,
                     y=c.imag,
                     fill="black",
                     content=str(int(x)) if x == int(x) else str(x),
                     text_anchor="middle",
                     dominant_baseline="auto",
                     font_size=12))
            for y in ys:
                if y != int(y):
                    continue
                c = transform_pt(plan_i, y*1j + boxes[plan_i][0].real)
                lines.append(tag_str(
                    "text",
                     x=c.real,
                     y=c.imag,
                     fill="black",
                     content=(str(int(y)) if y == int(y) else str(y)),
                     text_anchor="start",
                     alignment_baseline="middle",
                     font_size=12))

        lines.append("</svg>")
        return "\n".join(lines)
