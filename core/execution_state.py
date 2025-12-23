from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class ExecutionState:
    topic: str

    pipeline_stage: str = "initialized"

    artifacts: Dict[str, Any] = field(default_factory=dict)
    validations: Dict[str, Any] = field(default_factory=dict)

    timings: Dict[str, float] = field(default_factory=dict)

    def set_stage(self, stage: str):
        self.pipeline_stage = stage

    def add_artifact(self, name: str, value: Any):
        self.artifacts[name] = value

    def add_validation(self, name: str, value: Any):
        self.validations[name] = value
