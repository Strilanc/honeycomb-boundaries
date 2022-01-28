import csv
import dataclasses
import math
import pathlib
import threading
import time
from typing import Optional, Tuple, Dict, List, Callable, Union, Iterable, TypeVar

import numpy as np
import stim

from .decoding import sample_decode_count_correct
from .probability_util import log_binomial, binary_search

CSV_HEADER = ",".join([
    "data_width",
    "data_height",
    "rounds",
    "noise",
    "circuit_style",
    "preserved_observable",
    "code_distance",
    "num_qubits",
    "num_shots",
    "num_correct",
    "total_processing_seconds",
    "decoder",
    "version",
])
CSV_HEADER_VERSION = 2


@dataclasses.dataclass(frozen=True, unsafe_hash=True, order=True)
class DecodingProblemDesc:
    # noinspection PyUnresolvedReferences
    """Succinct data summarizing a decoding problem.

    Attributes:
        data_width: The width of the grid of data qubits.
        data_height: The height of the grid of data qubits.
        code_distance: Number of physical errors required to cause a logical error.
        num_qubits: Total number of noisy qubits in the system.
        rounds: The number of times measurement qubits are measured.
        noise: The strength of noise being applied.
        circuit_style: Names the circuit being run.
        preserved_observable: Names the observable being preserved against noise.
        decoder: The name of the decoder used to correct errors.
    """
    data_width: int
    data_height: int
    code_distance: int
    num_qubits: int
    rounds: int
    noise: float
    circuit_style: str
    preserved_observable: str
    decoder: str

    def with_changes(self,
                     *,
                     data_width: Optional[int] = None,
                     data_height: Optional[int] = None,
                     code_distance: Optional[int] = None,
                     num_qubits: Optional[int] = None,
                     rounds: Optional[int] = None,
                     noise: Optional[float] = None,
                     circuit_style: Optional[str] = None,
                     preserved_observable: Optional[str] = None,
                     decoder: Optional[str] = None,
    ) -> 'DecodingProblemDesc':
        return DecodingProblemDesc(
            data_width=self.data_width if data_width is None else data_width,
            data_height=self.data_height if data_height is None else data_height,
            code_distance=self.code_distance if code_distance is None else code_distance,
            num_qubits=self.num_qubits if num_qubits is None else num_qubits,
            rounds=self.rounds if rounds is None else rounds,
            noise=self.noise if noise is None else noise,
            circuit_style=self.circuit_style if circuit_style is None else circuit_style,
            preserved_observable=self.preserved_observable if preserved_observable is None else preserved_observable,
            decoder=self.decoder if decoder is None else decoder,
        )


@dataclasses.dataclass
class DecodingProblem:
    # noinspection PyUnresolvedReferences
    """Defines a decoding problem to sample from.

    Attributes:
        desc: Identifying information about the problem.
        circuit_maker: Produces a stim circuit with annotated noise and detectors.
    """
    desc: DecodingProblemDesc
    circuit_maker: Callable[[], stim.Circuit]

    def sample_correct_count(self, shots: int) -> int:
        return sample_decode_count_correct(
            num_shots=shots,
            circuit=self.circuit_maker(),
            decoder=self.desc.decoder,
        )



class WorkManager:
    def __init__(self, num_threads: int):
        self.lock = threading.Lock()
        self.available = threading.Semaphore(value=num_threads)
        self.exceptions = []

    def has_failed(self) -> bool:
        with self.lock:
            return bool(self.exceptions)


def collect_simulated_experiment_data(problems: Iterable[DecodingProblem],
                                      *,
                                      start_batch_size: int,
                                      max_shots: int,
                                      max_batch_size: Optional[int] = None,
                                      max_errors: int,
                                      num_threads: int = 1,
                                      out_path: Optional[Union[str, pathlib.Path]],
                                      merge_mode: str):
    """
    Args:
        problems: The decoding problems to collect sample data from.
        start_batch_size: The minimum number of samples to take from each case.

            This property effectively controls the quality of estimates of error rates when the true
            error is close to 50%

        max_shots: The maximum cutoff number of samples to take from each case.

            This property effectively controls the "noise floor" below which error rates cannot
            be estimated well. For example, setting this to 1e6 means that error rates below
            5e-5 will have estimates with large similar-relative-likelihood regions.

            This property overrides all the properties that ask for more samples until some
            statistical requirement is met.
        max_errors: More samples will be taken until the number of logical errors seen
            is at least this large. Set to 10 or 100 for fast estimates. Set to 1000 or 10000 for
            good statistical estimates of low probability errors.
        out_path: Where to write the CSV sample statistic data. Setting this to none doesn't write
            to file; only writes to stdout.
        max_batch_size: Defaults to unused. If set, then at most this many shots are collected at one
            time.
        merge_mode: Determines how new data and previous data are combined.
            "replace": Deletes previous data. Generates new data until the new data, by itself,
                meets the requested statistical requirements.
            "saturate": Generates new data until the new data, combined with previous data, meets
                the requested statistical requirements.
            "append": Ignores previous data. Generates new data until the new data, by itself, meets
                the requested statistical requirements.
        num_threads: Number of threads to use for parallel collection. This presumes you're using
            a C++ decoder that's safe to invoke separately across multiple threads.
    """
    print(CSV_HEADER, flush=True)
    previous_data = ProblemShotData({})
    if out_path is not None:
        if merge_mode == "replace" or not pathlib.Path(out_path).exists():
            with open(out_path, "w") as f:
                print(CSV_HEADER, file=f)
        if merge_mode == "saturate":
            previous_data = read_recorded_data(out_path)

    if max_batch_size is None:
        max_batch_size = max_shots

    lock = threading.Lock()
    def line_writer(line: str) -> None:
        with lock:
            if out_path is not None:
                with open(out_path, "a") as f:
                    print(line, file=f)
            print(line, flush=True)

    manager = WorkManager(num_threads=num_threads)
    try:
        for problem in problems:
            if manager.has_failed():
                break
            manager.available.acquire()
            threaded_collect_for_problem(
                problem=problem,
                prev_data=previous_data.data.get(problem.desc, ShotData()),
                start_batch_size=start_batch_size,
                max_batch_size=max_batch_size,
                max_errors=max_errors,
                max_shots=max_shots,
                line_writer=line_writer,
                manager=manager,
            )
        for k in range(num_threads):
            manager.available.acquire()
    except BaseException as ex:
        with manager.lock:
            manager.exceptions.append(ex)
        raise
    if manager.has_failed():
        raise manager.exceptions[0]


def threaded_collect_for_problem(*,
                                 problem: DecodingProblem,
                                 prev_data: 'ShotData',
                                 start_batch_size: int,
                                 max_batch_size: Optional[int],
                                 max_errors: int,
                                 max_shots: int,
                                 line_writer: Callable[[str], None],
                                 manager: WorkManager):
    def run():
        try:
            collect_for_problem(
                problem=problem,
                prev_data=prev_data,
                start_batch_size=start_batch_size,
                max_batch_size=max_batch_size,
                max_errors=max_errors,
                max_shots=max_shots,
                line_writer=line_writer,
                manager=manager,
            )
        except BaseException as ex:
            with manager.lock:
                manager.exceptions.append(ex)
            raise
        finally:
            manager.available.release()

    threading.Thread(target=run).start()


def collect_for_problem(*,
                        problem: DecodingProblem,
                        prev_data: 'ShotData',
                        start_batch_size: int,
                        max_batch_size: Optional[int],
                        max_errors: int,
                        max_shots: int,
                        line_writer: Callable[[str], None],
                        manager: WorkManager):
    new_data = prev_data.copy()
    num_next_shots = start_batch_size
    if max_batch_size is not None and max_batch_size < start_batch_size:
        num_next_shots = max_batch_size
    while new_data.num_errors < max_errors and new_data.num_shots < max_shots and not manager.has_failed():
        num_next_shots = min(num_next_shots, max_batch_size, max_shots - new_data.num_shots)

        t0 = time.monotonic()
        num_correct = sample_decode_count_correct(
            num_shots=num_next_shots,
            circuit=problem.circuit_maker(),
            decoder=problem.desc.decoder,
        )
        t1 = time.monotonic()
        record = ",".join(str(e) for e in [
            problem.desc.data_width,
            problem.desc.data_height,
            problem.desc.rounds,
            problem.desc.noise,
            problem.desc.circuit_style,
            problem.desc.preserved_observable,
            problem.desc.code_distance,
            problem.desc.num_qubits,
            num_next_shots,
            num_correct,
            t1 - t0,
            problem.desc.decoder,
            CSV_HEADER_VERSION,
        ])
        if manager.has_failed():
            # Don't write more results after KeyboardInterrupt.
            return
        line_writer(record)

        new_data.num_shots += num_next_shots
        new_data.num_correct += num_correct
        new_data.total_processing_seconds += t1 - t0
        num_next_shots *= 2


def collect_detection_fraction_data(problems: Iterable[DecodingProblem],
                                    *,
                                    shots: int,
                                    out_path: Optional[Union[str, pathlib.Path]],
                                    discard_previous_data: bool):
    print(CSV_HEADER, flush=True)
    if out_path is not None:
        if discard_previous_data or not pathlib.Path(out_path).exists():
            with open(out_path, "w") as f:
                print(CSV_HEADER, file=f)

    for problem in problems:
        t0 = time.monotonic()
        samples = problem.circuit_maker().compile_detector_sampler().sample(shots)
        num_detections = np.count_nonzero(samples)
        num_samples = math.prod(samples.shape)
        t1 = time.monotonic()
        record = ",".join(str(e) for e in [
            problem.desc.data_width,
            problem.desc.data_height,
            problem.desc.rounds,
            problem.desc.noise,
            problem.desc.circuit_style,
            "-",
            problem.desc.code_distance,
            problem.desc.num_qubits,
            num_samples,
            num_samples - num_detections,
            t1 - t0,
            "detection_fraction",
            CSV_HEADER_VERSION,
        ])
        if out_path is not None:
            with open(out_path, "a") as f:
                print(record, file=f)
        print(record, flush=True)


@dataclasses.dataclass
class RemainingWork:
    shot_data: 'ShotData'
    max_shots: int
    max_errors: int
    threshold_circuit_breaker: float

    @property
    def finished(self) -> bool:
        if self.shot_data.num_shots >= self.max_shots:
            return True
        if self.shot_data.num_errors >= self.max_errors:
            return True
        if self.shot_data.logical_error_rate >= self.threshold_circuit_breaker and self.shot_data.num_shots >= 10:
            return True
        return False

    @property
    def remaining_shots(self) -> int:
        if self.finished:
            return 0
        return self.max_shots - self.shot_data.num_shots

    @property
    def remaining_errors(self) -> int:
        if self.finished:
            return 0
        return self.max_errors - self.shot_data.num_errors

    @property
    def remaining_time(self) -> float:
        if self.finished:
            return 0
        times = [float('inf')]
        if self.shot_data.num_shots:
            times.append(self.remaining_shots * self.shot_data.total_processing_seconds / self.shot_data.num_shots)
        if self.shot_data.num_errors:
            times.append(self.remaining_errors * self.shot_data.total_processing_seconds / self.shot_data.num_errors)
        return min(times)



@dataclasses.dataclass
class ShotData:
    num_shots: int = 0
    num_correct: int = 0
    total_processing_seconds: float = 0

    def copy(self) -> 'ShotData':
        return ShotData(num_shots=self.num_shots,
                        num_correct=self.num_correct,
                        total_processing_seconds=self.total_processing_seconds)

    @property
    def num_errors(self) -> int:
        return self.num_shots - self.num_correct

    def remaining_work(self, max_shots: int, max_errors: int, threshold_circuit_breaker: float) -> RemainingWork:
        return RemainingWork(shot_data=self, max_shots=max_shots, max_errors=max_errors, threshold_circuit_breaker=threshold_circuit_breaker)

    def likely_error_rate_bounds(self, *, desired_ratio_vs_max_likelihood: float) -> Tuple[float, float]:
        """Compute relative-likelihood bounds.

        Returns the min/max error rates whose Bayes factors are within the given ratio of the maximum
        likelihood estimate.
        """
        actual_errors = self.num_shots - self.num_correct
        log_max_likelihood = log_binomial(p=actual_errors / self.num_shots, n=self.num_shots, hits=actual_errors)
        target_log_likelihood = log_max_likelihood + math.log(desired_ratio_vs_max_likelihood)
        acc = 100
        low = binary_search(
            func=lambda exp_err: log_binomial(p=exp_err / (acc * self.num_shots), n=self.num_shots, hits=actual_errors),
            target=target_log_likelihood,
            min_x=0,
            max_x=actual_errors * acc) / acc
        high = binary_search(
            func=lambda exp_err: -log_binomial(p=exp_err / (acc * self.num_shots), n=self.num_shots, hits=actual_errors),
            target=-target_log_likelihood,
            min_x=actual_errors * acc,
            max_x=self.num_shots * acc) / acc
        return low / self.num_shots, high / self.num_shots

    @property
    def logical_error_rate(self) -> float:
        if self.num_shots == 0:
            return 1
        return (self.num_shots - self.num_correct) / self.num_shots


TKey = TypeVar('TKey')


@dataclasses.dataclass
class ProblemShotData:
    data: Dict[DecodingProblemDesc, ShotData]

    def grouped_by(self,
                   key: Callable[[DecodingProblemDesc], TKey],
                   *,
                   reverse: bool = False) -> Dict[TKey, 'ProblemShotData']:
        groups = {}
        for k, v in self.data.items():
            group = key(k)
            groups.setdefault(group, ProblemShotData({})).data[k] = v
        return {k: groups[k] for k in sorted(groups.keys(), reverse=reverse)}

    def merged_total(self) -> 'ShotData':
        d = ShotData()
        for v in self.data.values():
            d.num_shots += v.num_shots
            d.num_correct += v.num_correct
            d.total_processing_seconds += v.total_processing_seconds
        return d

    def filter(self, predicate: Callable[[DecodingProblemDesc], bool]) -> 'ProblemShotData':
        result = ProblemShotData({})
        for k, v in self.data.items():
            if predicate(k):
                result.data[k] = v
        return result


def read_recorded_data(*paths: Union[str, pathlib.Path]) -> ProblemShotData:
    result = ProblemShotData({})
    for path in paths:
        with open(path, "r") as f:
            for row in csv.DictReader(f):
                key = DecodingProblemDesc(
                    code_distance=int(row["code_distance"]),
                    num_qubits=int(row["num_qubits"]),
                    data_width=int(row["data_width"]),
                    data_height=int(row["data_height"]),
                    rounds=int(row["rounds"]),
                    noise=float(row["noise"]),
                    circuit_style=row["circuit_style"],
                    preserved_observable=row["preserved_observable"],
                    decoder=row["decoder"],
                )
                val = result.data.setdefault(key, ShotData())
                val.num_shots += int(row["num_shots"])
                val.num_correct += int(row["num_correct"])
                val.total_processing_seconds += float(row["total_processing_seconds"])
    return result
