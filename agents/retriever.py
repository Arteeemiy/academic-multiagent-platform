from crewai import Agent


def create_retriever(llm):
    return Agent(
        role="Academic Retriever",
        goal=(
            "Retrieve strictly relevant academic context from the knowledge base "
            "for the given topic or subtask."
        ),
        backstory=(
            "You are a retrieval-only agent. "
            "You do NOT write prose and do NOT summarize unless explicitly asked. "
            "Your responsibility is to retrieve factual academic text chunks "
            "from the vector database using the provided retrieval tool."
        ),
        tools=[],  # tools будут переданы в Task
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )
