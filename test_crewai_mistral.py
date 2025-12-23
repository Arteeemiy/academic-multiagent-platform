from crewai import Agent, Task, Crew
from crewai.llm import LLM
import os

os.environ["LITELLM_LOGGING"] = "false"
os.environ["CREWAI_TRACING_ENABLED"] = "false"

# =========================
# CONFIG
# =========================
MISTRAL_API_KEY = "0tpGzK3spBkKDX57ssI9N5pJ62CNIRui"

# =========================
# LLM (MISTRAL)
# =========================
llm = LLM(
    model="mistral/mistral-large-2512",
    api_key=MISTRAL_API_KEY,
    base_url="https://api.mistral.ai/v1",
    max_tokens=1024,
    temperature=0.2,
)

# =========================
# AGENT
# =========================
agent = Agent(
    role="Analyst",
    goal="Answer questions clearly and concisely",
    backstory="You are a precise and thoughtful assistant.",
    llm=llm,
    verbose=True,
)

# =========================
# TASK
# =========================
task = Task(
    description=(
        "Explain in 3–5 sentences what CrewAI is used for "
        "and give one practical example."
    ),
    expected_output="A short clear explanation.",
    agent=agent,
)

# =========================
# CREW
# =========================
crew = Crew(agents=[agent], tasks=[task], verbose=True)

# =========================
# RUN
# =========================
if __name__ == "__main__":
    result = crew.kickoff()
    print("\n=== FINAL RESULT ===\n")
    print(result)
