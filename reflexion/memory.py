"""Episodic memory for Reflexion: stores verbal self-reflections per task.

In the paper this is the long-term memory that persists "lessons learned"
across trials, letting the Actor avoid repeating the same mistake. Kept
intentionally simple: an in-memory list per task id with most-recent-first
recall, serializable to JSON for longer runs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class EpisodicMemory:
    """Verbal lesson store, keyed by task id."""

    lessons: dict[str, list[str]] = field(default_factory=dict)

    def add(self, task_id: str, reflection: str) -> None:
        """Store one self-reflection for a task (newest first)."""
        self.lessons.setdefault(task_id, []).insert(0, reflection)

    def recall(self, task_id: str, k: int = 3) -> list[str]:
        """Return up to k most recent reflections for a task."""
        return self.lessons.get(task_id, [])[:k]

    def to_text(self, task_id: str, k: int = 3) -> str:
        """Render recalled lessons as prompt context for the Actor."""
        lessons = self.recall(task_id, k)
        if not lessons:
            return "(no prior lessons)"
        return "\n".join(f"- Lesson {i + 1}: {l}" for i, l in enumerate(lessons))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.lessons, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "EpisodicMemory":
        with open(path, encoding="utf-8") as fh:
            return cls(lessons=json.load(fh))
