from crewai import Task


def plan_task(agent, topic):
    return Task(
        description=f"""
You are planning an academic coursework.

Topic: {topic}

Return a structured outline with clear section titles and bullet points.
""".strip(),
        expected_output="Structured coursework outline",
        agent=agent,
    )


def write_intro_task(agent, topic, plan_task_ref):
    return Task(
        description=f"""
Write the INTRODUCTION section for an academic coursework.

Hard rules:
- Use ONLY the provided knowledge context
- Every factual statement MUST end with a citation like [source]
- Do NOT invent facts or sources
- If required information is missing, explicitly write:
  "Недостаточно данных в источниках."

Return ONLY the final introduction text (no commentary).

Topic: {topic}
""".strip(),
        expected_output="Introduction section with citations",
        agent=agent,
        context=[plan_task_ref],  # 👈 логический контекст (план)
    )


def validate_task(agent, intro_task_ref):
    return Task(
        description="""
Validate the introduction based on the provided knowledge context.

Validation rules:
- Every factual claim must be supported by the knowledge context
- All citations must exist and be consistent
- No hallucinated or speculative statements allowed

Return:
- PASS or FAIL
- List of detected problems (if any)
- Suggested fixes
""".strip(),
        expected_output="Validation report",
        agent=agent,
        context=[intro_task_ref],
    )


def edit_intro_task(agent, intro_task_ref, validation_task_ref):
    return Task(
        description="""
Improve the academic style and clarity of the introduction.

Rules:
- Preserve ALL citations exactly as they appear
- Do NOT add new factual content
- If validation report indicates FAIL, fix all issues first

Return ONLY the improved introduction text.
""".strip(),
        expected_output="Improved introduction",
        agent=agent,
        context=[intro_task_ref, validation_task_ref],
    )
