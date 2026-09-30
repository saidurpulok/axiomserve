"""Admission ordering is independent of batching and model execution."""

from typing import Protocol

from axiomserve.core.request import RequestState


class SchedulingPolicy(Protocol):
    def order(self, waiting: list[RequestState]) -> list[RequestState]: ...


class FCFS:
    def order(self, waiting: list[RequestState]) -> list[RequestState]:
        return sorted(waiting, key=lambda request: request.arrival_time)


class ShortestPromptFirst:
    def order(self, waiting: list[RequestState]) -> list[RequestState]:
        return sorted(waiting, key=lambda request: (len(request.input_ids), request.arrival_time))


def make_policy(name: str) -> SchedulingPolicy:
    if name == "fcfs":
        return FCFS()
    if name == "shortest_prompt":
        return ShortestPromptFirst()
    raise ValueError(f"unknown scheduler: {name}")
