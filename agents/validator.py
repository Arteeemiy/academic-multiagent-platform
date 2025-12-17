from crewai import Agent


def create_validator(llm):
    return Agent(
        role="Academic Validator",
        goal=(
            "Verify that all claims are fully supported by the provided knowledge context "
            "and that citations are accurate and complete"
        ),
        backstory=(
            "A strict academic reviewer who rejects any unsupported or speculative statements."
        ),
        llm=llm,
        verbose=True,
    )
