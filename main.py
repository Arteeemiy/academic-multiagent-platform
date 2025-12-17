from dotenv import load_dotenv
from pathlib import Path
import os

ENV_PATH = Path(
    "/Users/user/Desktop/AI_DL/LLM/MyProjects/AI_Multiagent_System_Coursework/.env"
)
load_dotenv(dotenv_path=ENV_PATH)

print("ENV PATH EXISTS:", ENV_PATH.exists())
print("KEY:", os.getenv("OPENAI_API_KEY"))

from crewai import LLM
from knowledge.loaders import load_knowledge_sources
from agents.planner import create_planner
from agents.writer import create_writer
from agents.editor import create_editor
from agents.validator import create_validator
from crew.crew import create_crew

import os

print("KEY:", os.getenv("OPENAI_API_KEY"))


def main():
    knowledge_sources = load_knowledge_sources()
    llm = LLM(
        model="mistral-large-2512",
        temperature=0,
        base_url="https://api.mistral.ai/v1",
    )

    agents = {
        "planner": create_planner(llm),
        "writer": create_writer(llm),
        "validator": create_validator(llm),
        "editor": create_editor(llm),
    }

    topic = "Разработка базы данных информационной подсистемы учета характеристик биологически опасных объектов в муниципальном образовании"
    crew = create_crew(agents, topic, knowledge_sources=knowledge_sources)

    result = crew.kickoff()
    print(result)


if __name__ == "__main__":
    main()
