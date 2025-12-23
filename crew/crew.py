from crewai import Crew, Process
from crew.tasks import (
    plan_task,
    retrieve_context_task,
    write_intro_task,
    validate_intro_task,
    edit_intro_task,
    write_chap1_task,
    validate_chap1_task,
    edit_chap1_task,
    write_chap2_task,
    validate_chap2_task,
    edit_chap2_task,
    write_conclusion_task,
    validate_conclusion_task,
    edit_conclusion_task,
)


def create_crew(agents, topic):
    plan = plan_task(agents["planner"], topic)
    retriever = retrieve_context_task(agents["retriever"], topic)
    chap1 = write_intro_task(agents["writer"], topic, plan, retriever)

    return Crew(
        agents=[agents["planner"], agents["writer"]],
        tasks=[plan, chap1],
        process=Process.sequential,
        verbose=True,
    )
