from abc import ABC, abstractmethod


class IntervalMethod(ABC):
    @abstractmethod
    def solve(
        self,
        interval_a: float,
        interval_b: float,
        tolerance: float,
        max_iterations: int,
        function_f: str,
        precision: int,
    ) -> dict:
        pass

    @abstractmethod
    def validate_input(
        interval_a: float,
        interval_b: float,
        tolerance: float,
        max_iterations: int,
        function_f: str,
    ) -> str | bool:
        pass
