"""Reflexion: language agents with verbal reinforcement learning.

A clean, runnable implementation of the Reflexion loop (Shinn et al., 2023):
an Actor produces ReAct-style trajectories, an Evaluator scores them, and a
Self-Reflection model turns failures into verbal lessons stored in episodic
memory for the next trial.
"""

from .agent import ReflexionAgent, TrialResult
from .llm import LLMBackend, MockBackend, OpenAICompatibleBackend
from .memory import EpisodicMemory

__all__ = [
    "ReflexionAgent",
    "TrialResult",
    "LLMBackend",
    "MockBackend",
    "OpenAICompatibleBackend",
    "EpisodicMemory",
]
