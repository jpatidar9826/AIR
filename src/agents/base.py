from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class AgentStep:
    agent: str
    action: str
    output_summary: str
    seconds: float


@dataclass
class Trace:
    steps: list[AgentStep] = field(default_factory=list)

    def record(self, agent: str, action: str, summary: str, start: float):
        self.steps.append(AgentStep(agent, action, summary, round(time.time() - start, 3)))

    def as_rows(self):
        return [s.__dict__ for s in self.steps]
