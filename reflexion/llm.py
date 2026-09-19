"""LLM backends for the Reflexion agent.

The agent only needs two capabilities from a backend:
  * ``act``      - produce the next thought+action given the trajectory so far
  * ``reflect``  - produce a verbal lesson from a failed trajectory

``MockBackend`` is a deterministic scripted backend for the bundled toy task,
so the demo runs offline and visibly improves across trials. Swap in
``OpenAICompatibleBackend`` (or your own subclass) for real models.
"""
from __future__ import annotations

import json
import os
import urllib.request


class LLMBackend:
    """Interface the Reflexion agent programs against."""

    def act(self, context: str) -> str:
        """Return the next step as ``Thought: ...\\nAction: tool[arg]``."""
        raise NotImplementedError

    def reflect(self, context: str) -> str:
        """Return a short verbal lesson learned from a failed trajectory."""
        raise NotImplementedError


class MockBackend(LLMBackend):
    """Deterministic scripted backend for the toy Wikipedia QA task.

    Trial 0 behaves like a naive agent: it looks up the ambiguous entity
    "The Prestige" and lands on the novel instead of the 2006 film, so it
    answers 1941 (Christopher Priest's birth year) and fails.

    From trial 1 on, once episodic memory holds a reflection mentioning
    disambiguation, it searches precisely ("The Prestige 2006 film") and
    answers 1970 (Christopher Nolan's birth year).
    """

    def __init__(self) -> None:
        self._trial = 0
        self._step = 0
        self._memory_text = ""

    def new_trial(self, trial: int, memory_text: str) -> None:
        self._trial = trial
        self._step = 0
        self._memory_text = memory_text.lower()

    def act(self, context: str) -> str:
        learned = "disambiguat" in self._memory_text
        step = self._step
        self._step += 1
        if not learned:
            if step == 0:
                return "Thought: I need the director of 'The Prestige'.\nAction: search[The Prestige]"
            if step == 1:
                return "Thought: The first result looks right, I'll look it up.\nAction: lookup[The Prestige]"
            return "Thought: Christopher Priest was born in 1941.\nAction: finish[1941]"
        if step == 0:
            return (
                "Thought: Last time I hit the novel. This time I'll disambiguate "
                "with the year and 'film'.\nAction: search[The Prestige 2006 film]"
            )
        return "Thought: The 2006 film was directed by Christopher Nolan, born 1970.\nAction: finish[1970]"

    def reflect(self, context: str) -> str:
        return (
            "I was unsuccessful because I looked up the ambiguous entity "
            "'The Prestige' and retrieved the novel instead of the 2006 film. "
            "Next time I should disambiguate entities by including the year "
            "and 'film' in the search query."
        )


class OpenAICompatibleBackend(LLMBackend):
    """Backend for any OpenAI-compatible chat-completions endpoint.

    Set ``OPENAI_API_KEY`` and optionally ``OPENAI_BASE_URL``
    (defaults to https://api.openai.com/v1) and ``REFLEXION_MODEL``.
    """

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get("REFLEXION_MODEL", "gpt-4o-mini")
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

    def _chat(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.load(resp)
        return body["choices"][0]["message"]["content"].strip()

    def act(self, context: str) -> str:
        return self._chat(
            "You are a ReAct agent. Reply with exactly two lines: "
            "'Thought: ...' then 'Action: tool[argument]'. "
            "Tools: search[query], lookup[entity], finish[answer].",
            context,
        )

    def reflect(self, context: str) -> str:
        return self._chat(
            "You are a self-reflection module. Given a failed agent trajectory "
            "and evaluator feedback, write 1-2 sentences: what went wrong and "
            "what to do differently next time. Be specific and actionable.",
            context,
        )
