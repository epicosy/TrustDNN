from dataclasses import dataclass
from pathlib import Path
from safednn.core.dataset.base import Dataset
from safednn.core.model import Model


@dataclass
class Execution:
    timestamp: int
    output: Path
    executed: bool
    status: str
    duration: float
    mem_usage: float
    return_code: int

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "executed": self.executed,
            "status": self.status,
            "duration": self.duration,
            "mem_usage": self.mem_usage,
            "return_code": self.return_code,
            "output": self.output
        }


@dataclass
class Instance:
    dataset: Dataset
    model: Model
    working_dir: Path
    phase: str

    def __str__(self):
        return f"<Instance: {self.phase} - {self.working_dir} - {self.model.name} - {self.dataset.name}>"
