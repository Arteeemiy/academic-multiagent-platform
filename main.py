# main.py
from dotenv import load_dotenv
from pathlib import Path
import os
import asyncio
import json

# =====================
# ENV
# =====================
ENV_PATH = Path(
    "/Users/user/Desktop/AI_DL/LLM/MyProjects/AI_Multiagent_System_Coursework/.env"
)
load_dotenv(dotenv_path=ENV_PATH)

assert os.getenv("MISTRAL_API_KEY"), "MISTRAL_API_KEY not found in env"

os.environ["CREWAI_TELEMETRY_ENABLED"] = "false"
os.environ["CREWAI_DISABLE_OPENAI"] = "true"


# =====================
# PIPELINE
# =====================
from core.execution_state import ExecutionState
from core.crew_factory import run_coursework_crew


TOPIC = (
    "Разработка базы данных информационной подсистемы учета характеристик "
    "биологически опасных объектов в муниципальном образовании"
)


async def run():
    # workflow-level state (НЕ user state)
    exec_state = ExecutionState(topic=TOPIC)

    result = await run_coursework_crew(
        topic=TOPIC,
        execution_state=exec_state,
    )

    print("\n===== FINAL RESULT =====\n")
    print(result)

    print("\n===== PIPELINE STATE =====\n")
    print(f"Stage: {exec_state.pipeline_stage}")
    print(f"Timings: {exec_state.timings}")
    print(f"Artifacts: {list(exec_state.artifacts.keys())}")

    # опционально — сохранить state для дебага
    with open("outputs/execution_state.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "topic": exec_state.topic,
                "pipeline_stage": exec_state.pipeline_stage,
                "timings": exec_state.timings,
                "artifacts": list(exec_state.artifacts.keys()),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    asyncio.run(run())
