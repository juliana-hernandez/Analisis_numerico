from abc import ABC, abstractmethod


class IterativeMethod(ABC):
    @abstractmethod
    def solve(
        self,
        x0: float,
        tolerance: float,
        max_iterations: int,
        precision: int,
        function_f: str,
        **kwargs,
    ) -> dict:
        pass

    @abstractmethod
    def validate_input(
        self,
        x0: float,
        tolerance: float,
        max_iterations: int,
        function_f: str,
        **kwargs,
    ) -> str | bool:
        pass
