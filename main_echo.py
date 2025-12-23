from dotenv import load_dotenv
from pathlib import Path
import os
import litellm
import os

os.environ["LITELLM_LOGGING"] = "false"
os.environ["LITELLM_ENABLE_JSON_LOGGING"] = "false"
os.environ["LITELLM_CALLBACKS"] = "[]"
os.environ["LITELLM_DISABLE_PROXY"] = "true"
os.environ["LITELLM_COLD_STORAGE"] = "false"
os.environ["CREWAI_TELEMETRY_ENABLED"] = "false"


os.environ["CREWAI_DISABLE_OPENAI"] = "true"


# =====================
# ENV
# =====================
ENV_PATH = Path(
    "/Users/user/Desktop/AI_DL/LLM/MyProjects/AI_Multiagent_System_Coursework/.env"
)
load_dotenv(dotenv_path=ENV_PATH)

assert os.getenv("MISTRAL_API_KEY"), "MISTRAL_API_KEY not found in env"

# =====================
# CrewAI
# =====================
from crewai.llm import LLM
from agents.planner import create_planner
from agents.writer import create_writer
from agents.editor import create_editor
from agents.validator import create_validator
from agents.retriever import create_retriever_agent
from crew.crew import create_crew


def main():
    os.environ["LITELLM_NUM_RETRIES"] = "3"
    os.environ["LITELLM_RETRY_DELAY"] = "2"

    llm = LLM(
        model="mistral/mistral-medium-latest",
        temperature=0.3,
        timeout=60,
        max_tokens=2048,
        max_retries=5,
        callbacks=[],  # 🔥 ВАЖНО
    )

    agents = {
        "planner": create_planner(llm),
        "writer": create_writer(llm),
        "validator": create_validator(llm),
        "editor": create_editor(llm),
        "retriever": create_retriever_agent(llm),
    }

    topic = (
        "Разработка базы данных информационной подсистемы учета характеристик "
        "биологически опасных объектов в муниципальном образовании"
    )

    crew = create_crew(agents, topic)
    result = crew.kickoff()

    print("\n===== FINAL RESULT =====\n")
    print(result)


if __name__ == "__main__":
    main()
