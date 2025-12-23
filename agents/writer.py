from crewai import Agent


def create_writer(llm):
    return Agent(
        role="Academic Writer",
        goal=(
            "Write coursework sections strictly grounded in the provided knowledge context, "
            "with clear citations for every factual claim"
        ),
        backstory=(
            "An experienced academic writer who never hallucinates facts and "
            "explicitly states when information is missing from the sources."
        ),
        llm=llm,
        verbose=False,
    )
