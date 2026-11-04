"""Small integration envelope: timing must travel with compiled choices."""
from dataclasses import dataclass, field
from Schema import CompiledProblem


@dataclass
class Compilation:
    problem: CompiledProblem
    slot_durations: list[float]
    warnings: list[str] = field(default_factory=list)


class PipelineError(ValueError):
    """Only explicit, non-sensitive integration errors may reach the website."""
    def __init__(self, message: str):
        super().__init__(message)
        self.public_message = message
