"""Toy tasks with scripted tools and heuristic evaluators.

These stand in for the paper's HotpotQA / ALFWorld / programming benchmarks:
small, deterministic environments where the Reflexion mechanism (fail ->
verbal lesson -> retry with memory) is observable without API keys.

``ToyWikiQATask`` mirrors the HotpotQA setup: a multi-hop-style question over
a mock Wikipedia where one entity name is ambiguous (film vs. novel). The
heuristic evaluator plays the paper's "exact match + feedback" role.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToyWikiQATask:
    """'What is the birth year of the director of the film The Prestige (2006)?'

    The trap: ``lookup[The Prestige]`` returns the *novel* entry. The agent
    must learn (via self-reflection) to disambiguate with the year + 'film'.
    """

    id: str = "prestige-director-birth-year"
    question: str = "What is the birth year of the director of the film The Prestige (2006)?"
    expected: str = "1970"
    success_threshold: float = 1.0
    tools: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.tools = {
            "search": self._search,
            "lookup": self._lookup,
        }

    # -- mock Wikipedia -----------------------------------------------------
    @staticmethod
    def _search(query: str) -> str:
        q = query.lower()
        if "2006" in q and "film" in q:
            return "1. The Prestige (2006 film) directed by Christopher Nolan (born 1970)."
        return (
            "1. The Prestige (novel), a 1995 novel by Christopher Priest (born 1941). "
            "2. The Prestige (2006 film) directed by Christopher Nolan (born 1970)."
        )

    @staticmethod
    def _lookup(entity: str) -> str:
        if entity.strip().lower() == "the prestige":
            # Ambiguous: returns the novel entry, like a naive top-1 lookup.
            return "The Prestige is a 1995 novel by British writer Christopher Priest (born 1941)."
        return f"No entry found for '{entity}'."

    # -- evaluator ----------------------------------------------------------
    def evaluate(self, answer: str | None, steps) -> tuple[float, str]:
        if answer is None:
            return 0.0, "No answer produced. Use finish[answer] to submit."
        if answer.strip() == self.expected:
            return 1.0, "Correct."
        if answer.strip() == "1941":
            return 0.0, (
                "Incorrect. 1941 is the birth year of Christopher Priest, the "
                "novelist — you looked up the novel, not the 2006 film. The "
                "entity 'The Prestige' was ambiguous."
            )
        return 0.0, f"Incorrect. Expected {self.expected}, got {answer}."
