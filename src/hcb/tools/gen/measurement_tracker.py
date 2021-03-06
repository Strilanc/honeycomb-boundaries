"""This file contains a helper class for indexing measurements for stim."""

import collections
from typing import List, Any, FrozenSet, Iterable, DefaultDict, Optional, TypeVar, Generic, Union

import stim


T = TypeVar("T")


class Prev(Generic[T]):
    """A special class that indicates 'the previous measurement' to MeasurementTracker."""
    def __init__(self, v: Union[T, 'Prev[T]'], offset: int = 1):
        self.v: T
        self.offset: int
        if isinstance(v, Prev):
            self.v = v.v
            self.offset = offset + v.offset
        else:
            self.v = v
            self.offset = offset

    def __eq__(self, other):
        if not isinstance(other, Prev):
            return NotImplemented
        return self.v == other.v and self.offset == other.offset

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash((self.v, self.offset))

    def __add__(self, other: Any) -> 'Prev[Any]':
        return Prev(self.v + other, self.offset)

    def __repr__(self):
        return f"Prev({self.v!r}, offset={self.offset!r})"


class MeasurementTracker:
    """Tracks measurements and groups of measurements, for producing stim record targets."""
    def __init__(self):
        self.history: DefaultDict[Any, List[Optional[FrozenSet[int]]]] = collections.defaultdict(list)
        self.t = 0

    def add_measurements(self, *keys: Any):
        for key in keys:
            assert key is not None
            assert not isinstance(key, Prev)
            self.history[key].append(frozenset([self.t]))
            self.t += 1

    def add_dummies(self, *keys: Any, obstacle: bool = False):
        v = None if obstacle else frozenset()
        for key in keys:
            assert key is not None
            assert not isinstance(key, Prev)
            self.history[key].append(v)

    def add_group(self, *keys: Any, group_key: Any):
        assert not isinstance(group_key, Prev)
        assert group_key is not None
        self.history[group_key].append(self.measurement_time_set(*keys))

    def measurement_time_set(self, *keys: Any) -> Optional[FrozenSet[int]]:
        result = frozenset()
        for k in keys:
            t = 1
            if isinstance(k, Prev):
                t += k.offset
                k = k.v
            h = self.history[k]
            assert t <= len(h), f"Didn't add a dummy or an obstacle for {k!r}"
            v = self.history[k][-t]
            if v is None:
                return None
            result ^= v
        return result

    def get_record_targets(self,
                        *keys: Any,
                        for_time_after_measurement: Any = None) -> Optional[List[int]]:
        t0 = self.t
        if for_time_after_measurement is not None:
            t0, = self.history[for_time_after_measurement][-1]
            t0 += 1
        times = self.measurement_time_set(*keys)
        if times is None:
            return None
        return [stim.target_rec(t - t0) for t in sorted(times)]

    def append_detector(self,
                        *keys: Any,
                        out_circuit: stim.Circuit,
                        for_time_after_measurement: Any = None,
                        coords: Iterable[float] = ()):
        targets = self.get_record_targets(*keys, for_time_after_measurement=for_time_after_measurement)
        if targets is None:
            return
        out_circuit.append("DETECTOR", targets, coords)
