"""Tests for the Reflexion loop, memory, and toy task evaluator."""
from reflexion import EpisodicMemory, MockBackend, ReflexionAgent
from reflexion.agent import parse_action
from reflexion.tasks import ToyWikiQATask


def test_memory_recall_order():
    mem = EpisodicMemory()
    mem.add("t1", "first lesson")
    mem.add("t1", "second lesson")
    assert mem.recall("t1") == ["second lesson", "first lesson"]
    assert mem.recall("unknown") == []
    assert "(no prior lessons)" in mem.to_text("unknown")


def test_parse_action():
    assert parse_action("Thought: x\nAction: search[The Prestige]") == ("search", "The Prestige")
    assert parse_action("Action: finish[1970]") == ("finish", "1970")


def test_evaluator_accepts_correct_answer():
    task = ToyWikiQATask()
    reward, feedback = task.evaluate("1970", [])
    assert reward == 1.0


def test_evaluator_explains_ambiguity_trap():
    task = ToyWikiQATask()
    reward, feedback = task.evaluate("1941", [])
    assert reward == 0.0
    assert "ambiguous" in feedback


def test_reflexion_improves_across_trials():
    """The core claim: trial 0 fails, reflection is stored, trial 1 succeeds."""
    task = ToyWikiQATask()
    agent = ReflexionAgent(backend=MockBackend(), max_trials=3)
    result = agent.run(task)
    assert result.success
    assert result.answer == "1970"
    assert result.trial == 1  # failed once, then succeeded
    assert len(agent.memory.recall(task.id)) == 1
    assert "disambiguate" in agent.memory.recall(task.id)[0]
