from crewai import Agent


def create_planner(llm):
    return Agent(
        role="Academic Planner",
        goal="Break down coursework writing into clear, logical steps",
        backstory="Expert in academic project planning and structured reasoning.",
        llm=llm,
        verbose=True,
    )
