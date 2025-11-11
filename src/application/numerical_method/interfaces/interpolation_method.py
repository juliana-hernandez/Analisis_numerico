from abc import ABC, abstractmethod


class InterpolationMethod(ABC):
    @abstractmethod
    def solve(
        self,
        x: list[float],
        y: list[float],
    ) -> dict:
        pass

    @abstractmethod
    def validate_input(
        self,
        x_input: str,
        y_input: str,
    ) -> str | list[tuple[float, float]]:
        pass
