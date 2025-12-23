# core/crew_factory.py
import asyncio
import time
from crewai.llm import LLM

from agents.planner import create_planner
from agents.retriever import create_retriever
from agents.writer import create_writer
from agents.editor import create_editor
from agents.validator import create_validator
from crew.crew import create_crew

from core.execution_state import ExecutionState


async def run_coursework_crew(
    topic: str,
    execution_state: ExecutionState | None = None,
):
    """
    Runs full coursework multi-agent pipeline.

    execution_state:
        Optional workflow-level state container.
        Allows tracking stages, artifacts, timings.
    """

    exec_state = execution_state or ExecutionState(topic=topic)

    exec_state.set_stage("llm_init")
    start_ts = time.time()

    llm = LLM(
        model="mistral/mistral-medium-latest",
        temperature=0.3,
        timeout=60,
        max_tokens=2048,
        max_retries=3,
    )

    agents = {
        "planner": create_planner(llm),
        "retriever": create_retriever(llm),
        "writer": create_writer(llm),
        "validator": create_validator(llm),
        "editor": create_editor(llm),
    }

    exec_state.set_stage("crew_init")
    crew = create_crew(agents, topic)

    exec_state.set_stage("crew_running")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, crew.kickoff)

    exec_state.set_stage("crew_finished")

    exec_state.timings["total_sec"] = round(time.time() - start_ts, 2)
    exec_state.add_artifact("final_result", result)

    return result
