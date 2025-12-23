from crewai import Crew, Process
from crew.tasks import (
    plan_task,
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
    # =========================
    # PLANNING
    # =========================
    plan = plan_task(agents["planner"], topic)

    # =========================
    # INTRODUCTION
    # =========================
    intro = write_intro_task(agents["writer"], topic, plan)
    intro_val = validate_intro_task(agents["validator"], intro)
    intro_edit = edit_intro_task(agents["editor"], intro, intro_val)

    # =========================
    # CHAPTER 1
    # =========================
    chap1 = write_chap1_task(agents["writer"], topic, plan)
    chap1_val = validate_chap1_task(agents["validator"], chap1)
    chap1_edit = edit_chap1_task(agents["editor"], chap1, chap1_val)

    # =========================
    # CHAPTER 2
    # =========================
    chap2 = write_chap2_task(agents["writer"], topic, plan, chap1)
    chap2_val = validate_chap2_task(agents["validator"], chap2)
    chap2_edit = edit_chap2_task(agents["editor"], chap2, chap2_val)

    # =========================
    # CONCLUSION
    # =========================
    conclusion = write_conclusion_task(
        agents["writer"],
        topic,
        plan,
        intro_edit,
        chap1_edit,
        chap2_edit,
    )
    conclusion_val = validate_conclusion_task(agents["validator"], conclusion)
    conclusion_edit = edit_conclusion_task(
        agents["editor"],
        conclusion,
        conclusion_val,
    )

    # =========================
    # CREW
    # =========================
    return Crew(
        agents=list(agents.values()),
        tasks=[
            plan,
            intro,
            intro_val,
            intro_edit,
            chap1,
            chap1_val,
            chap1_edit,
            chap2,
            chap2_val,
            chap2_edit,
            conclusion,
            conclusion_val,
            conclusion_edit,
        ],
        process=Process.sequential,
        verbose=True,
    )
