"""The Reflexion agent: Actor + Evaluator + Self-Reflection loop.

Paper mapping (Shinn et al., 2023, "Reflexion: Language Agents with Verbal
Reinforcement Learning"):
  * Actor            -> ``_run_trial``: ReAct-style thought/action/observation loop
  * Evaluator        -> ``task.evaluate``: returns (reward, feedback); in the paper
                        this is a heuristic, an LLM, or self-evaluation depending
                        on the domain
  * Self-Reflection  -> ``backend.reflect``: turns (trajectory, feedback) into a
                        verbal lesson stored in episodic memory

The loop runs up to ``max_trials``; each trial conditions on the lessons of all
previous trials, which is what lets the agent improve without weight updates.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .llm import LLMBackend
from .memory import EpisodicMemory


@dataclass
class Step:
    thought: str
    action: str
    observation: str


@dataclass
class TrialResult:
    trial: int
    success: bool
    reward: float
    answer: str | None
    steps: list[Step] = field(default_factory=list)
    reflection: str | None = None


def parse_action(text: str) -> tuple[str, str]:
    """Parse ``Action: tool[argument]`` into (tool, argument)."""
    line = next((ln for ln in text.splitlines() if ln.strip().lower().startswith("action:")), "")
    payload = line.split(":", 1)[1].strip() if ":" in line else ""
    if "[" in payload and payload.endswith("]"):
        tool, arg = payload[:-1].split("[", 1)
        return tool.strip(), arg.strip()
    return payload, ""


class ReflexionAgent:
    def __init__(
        self,
        backend: LLMBackend,
        memory: EpisodicMemory | None = None,
        max_steps: int = 6,
        max_trials: int = 3,
    ) -> None:
        self.backend = backend
        self.memory = memory or EpisodicMemory()
        self.max_steps = max_steps
        self.max_trials = max_trials
        self.trial_history: list[TrialResult] = []

    def _run_trial(self, task, trial: int) -> TrialResult:
        memory_text = self.memory.to_text(task.id)
        if hasattr(self.backend, "new_trial"):
            self.backend.new_trial(trial, memory_text)

        steps: list[Step] = []
        context = (
            f"Task: {task.question}\n"
            f"Lessons from previous trials:\n{memory_text}\n"
            f"Tools: {', '.join(task.tools)}\n"
        )
        answer: str | None = None
        for _ in range(self.max_steps):
            history = context + "\n".join(
                f"Thought: {s.thought}\nAction: {s.action}\nObservation: {s.observation}"
                for s in steps
            )
            raw = self.backend.act(history + "\nNext step:")
            thought = next(
                (ln.split(":", 1)[1].strip() for ln in raw.splitlines()
                 if ln.strip().lower().startswith("thought:")),
                "",
            )
            tool, arg = parse_action(raw)
            if tool == "finish":
                answer = arg
                steps.append(Step(thought, f"finish[{arg}]", "done"))
                break
            observation = task.tools.get(tool, lambda a: f"unknown tool: {tool}")(arg)
            steps.append(Step(thought, f"{tool}[{arg}]", observation))

        reward, feedback = task.evaluate(answer, steps)
        return TrialResult(
            trial=trial,
            success=reward >= task.success_threshold,
            reward=reward,
            answer=answer,
            steps=steps,
        )

    def run(self, task) -> TrialResult:
        """Run Reflexion trials until success or ``max_trials`` is reached."""
        last: TrialResult | None = None
        for trial in range(self.max_trials):
            result = self._run_trial(task, trial)
            self.trial_history.append(result)
            last = result
            if result.success:
                return result
            reflection = self.backend.reflect(
                f"Task: {task.question}\n"
                f"Trajectory: {[ (s.action, s.observation) for s in result.steps ]}\n"
                f"Evaluator feedback: {task.evaluate(result.answer, result.steps)[1]}"
            )
            result.reflection = reflection
            self.memory.add(task.id, reflection)
        assert last is not None
        return last
