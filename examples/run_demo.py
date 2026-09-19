"""Run the Reflexion demo on the toy Wikipedia QA task.

Expected output: trial 0 fails (answers 1941, the novelist's birth year),
the agent reflects on the ambiguity, stores a verbal lesson, and trial 1
succeeds (1970) — the Reflexion loop working without any weight updates.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reflexion import EpisodicMemory, MockBackend, ReflexionAgent
from reflexion.tasks import ToyWikiQATask


def main() -> None:
    task = ToyWikiQATask()
    agent = ReflexionAgent(backend=MockBackend(), memory=EpisodicMemory(), max_trials=3)
    agent.run(task)

    print(f"Question: {task.question}\n")
    for r in agent.trial_history:
        status = "SUCCESS" if r.success else "FAIL"
        print(f"--- trial {r.trial}: {status} (answer={r.answer})")
        for s in r.steps:
            print(f"  {s.action}\n    -> {s.observation[:90]}")
        if r.reflection:
            print(f"  reflection: {r.reflection}\n")

    final = agent.trial_history[-1]
    print(f"Final answer: {final.answer} (expected {task.expected})")


if __name__ == "__main__":
    main()
