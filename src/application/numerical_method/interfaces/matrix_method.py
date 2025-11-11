from abc import ABC, abstractmethod


class MatrixMethod(ABC):
    @abstractmethod
    def solve(
        self,
        A: list[list[float]],
        b: list[float],
        x0: list[float],
        tolerance: float,
        max_iterations: int,
        **kwargs,
    ) -> dict:
        pass

    @abstractmethod
    def validate_input(
        self,
        matrix_a_raw: str,
        vector_b_raw: str,
        initial_guess_raw: str,
        tolerance: float,
        max_iterations: int,
        **kwargs,
    ) -> str | list:
        pass
