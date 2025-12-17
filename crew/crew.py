from crewai import Crew, Process
from crew.tasks import (
    plan_task,
    write_intro_task,
    validate_task,
    edit_intro_task,
)


def create_crew(agents, topic, knowledge_sources=None):
    t1 = plan_task(agents["planner"], topic)
    t2 = write_intro_task(agents["writer"], topic, t1)
    t3 = validate_task(agents["validator"], t2)
    t4 = edit_intro_task(agents["editor"], t2, t3)

    return Crew(
        agents=list(agents.values()),
        tasks=[t1, t2, t3, t4],
        process=Process.sequential,
        verbose=True,
        knowledge_sources=knowledge_sources,
    )
