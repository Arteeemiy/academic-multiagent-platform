from crewai import Agent


def create_editor(llm):
    return Agent(
        role="Academic Editor",
        goal="Improve clarity, coherence, and academic tone without altering factual content or citations",
        backstory="A professional academic editor focused on clarity and formal writing standards.",
        llm=llm,
        verbose=True,
    )
