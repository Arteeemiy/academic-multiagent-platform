from crewai import Task
from tools.retriever_tool import ChromaRetrieverTool


# ======================================================
# RETRIEVAL TASKS
# ======================================================


def retrieve_context_task(agent, topic: str, k: int = 9):
    return Task(
        description=f"""
Retrieve academic factual context for the topic.

STRICT RULES:
1. FIRST use the document_retriever tool (local knowledge base)
2. ONLY IF local sources return "Недостаточно данных в источниках."
   THEN use website_search tool
3. Return ONLY raw factual text
4. Do NOT summarize or explain
5. If both sources fail, return exactly:
   "Недостаточно данных в источниках."

TOPIC:
{topic}
""".strip(),
        expected_output=(
            "Raw factual academic text retrieved from sources, "
            "or 'Недостаточно данных в источниках.'"
        ),
        agent=agent,
        tools=[
            ChromaRetrieverTool(),
        ],
    )


# ======================================================
# PLANNING
# ======================================================


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


# ======================================================
# WRITING TASKS (RAG ENABLED)
# ======================================================


def write_intro_task(agent, topic, plan_task_ref, retrieved_context_ref):
    return Task(
        description=f"""
Write the INTRODUCTION section for an academic coursework.

STRICT RULES:
- Use ONLY the provided context
- Every factual statement MUST end with [source]
- Do NOT invent facts or sources
- Do NOT mention chunk IDs
- Missing info → "Недостаточно данных в источниках."

Return ONLY the final introduction text.

TOPIC:
{topic}
""".strip(),
        expected_output="Introduction section with citations",
        agent=agent,
        context=[plan_task_ref, retrieved_context_ref],
        output_file="outputs/intro.txt",
    )


def write_chap1_task(agent, topic, plan_task_ref, retrieved_task_ref):
    return Task(
        description=f"""
You are writing CHAPTER 1 of an academic coursework.

TOPIC:
{topic}

SOURCE RULE:
Use ONLY:
1) the provided plan
2) the retrieved context

CRITICAL:
- You have NO tools доступных. Никаких вызовов.
- If required info is missing, write exactly: "Недостаточно данных в источниках."

CITATIONS:
Every factual statement MUST end with [source].

MANDATORY STRUCTURE:
1. Requirements analysis
2. Conceptual design
3. Logical design
4. Description of entities and relationships

After section 4 output:
ER_DIAGRAM_CODE:
<valid DBML for dbdiagram.io>

DBML rules:
- Max 10 entities
- PK/FK required
""".strip(),
        expected_output="Complete Chapter 1 academic text with [source] + ER_DIAGRAM_CODE",
        agent=agent,
        context=[plan_task_ref, retrieved_task_ref],
        output_file="outputs/chap1.txt",
    )


def write_chap2_task(agent, topic, plan_task_ref, retrieved_task_ref):
    return Task(
        description=f"""
You are writing CHAPTER 2 of an academic coursework.

IMPORTANT EXECUTION RULE:
- You MAY use the document retriever internally to gather evidence.
- You MUST ALWAYS produce a final written chapter text.
- NEVER output tool calls, JSON, or retriever syntax in the final answer.
- The final answer must be pure academic text, ready to paste into the coursework.

CHAPTER THEME:
Database implementation and application development.

SOURCE CONSTRAINT:
- Use ONLY Chapter 1 and retrieved context.
- You MAY use the document retriever internally to gather evidence.
- You MUST ALWAYS produce a final written chapter text.
- NEVER output tool calls, JSON, or retriever syntax in the final answer.
- The final answer must be pure academic text, ready to paste into the coursework.

CHAPTER THEME:
Database implementation and application development.

SOURCE CONSTRAINT:
- Use ONLY Chapter 1 and retrieved context.
- Do NOT use general knowledge.
- Do NOT invent entities, technologies, standards, or regulations.
- Every factual statement MUST end with [source].
- Do NOT mention chunk IDs or retriever results.
- If information is missing, write exactly:
  "Недостаточно данных в источниках."

GLOBAL STRUCTURE:
1. DBMS selection justification
2. Database implementation
3. Tables and constraints
4. Relationships
5. Application development
6. Conclusions

TOPIC:
{topic}
""".strip(),
        expected_output="Complete Chapter 2 text with [source] citations",
        agent=agent,
        context=[plan_task_ref, retrieved_task_ref],
        output_file="outputs/chap2.txt",
    )


def write_conclusion_task(
    agent,
    topic,
    plan_task_ref,
    retrieved_task_ref,
):
    return Task(
        description=f"""
You are writing the CONCLUSION of an academic coursework.

RULES:
- Use ONLY previous sections
- Do NOT introduce new information
- No repetition
Every factual statement MUST be grounded in retrieved sources.
Use [source] as a citation placeholder.
Do NOT mention chunk IDs in the final text.
- Missing info → "Недостаточно данных в источниках."

CONTENT:
1. Achieved objectives
2. Key results
3. Advantages of the solution
4. Overall assessment

TOPIC:
{topic}
""".strip(),
        expected_output="Conclusion text",
        agent=agent,
        context=[
            plan_task_ref,
            retrieved_task_ref,
        ],
    )


# ======================================================
# VALIDATION TASKS (NO RAG)
# ======================================================


def validate_intro_task(agent, intro_task_ref):
    return Task(
        description="""
Validate the introduction.

Rules:
- All factual claims must be supported
- All citations must exist
- No hallucinations

Return:
- PASS or FAIL
- List of issues
- Suggested fixes
""".strip(),
        expected_output="Validation report",
        agent=agent,
        context=[intro_task_ref],
    )


def validate_chap1_task(agent, chap1_task_ref):
    return Task(
        description="""
Validate Chapter 1.

Rules:
- Factual correctness
- Citation consistency
- Logical coherence
""".strip(),
        expected_output="Validation report",
        agent=agent,
        context=[chap1_task_ref],
    )


def validate_chap2_task(agent, chap2_task_ref):
    return Task(
        description="""
Validate Chapter 2.

Rules:
- Compliance with Chapter 1
- No new entities or relations
- Citation correctness
""".strip(),
        expected_output="Validation report",
        agent=agent,
        context=[chap2_task_ref],
    )


def validate_conclusion_task(agent, conclusion_task_ref):
    return Task(
        description="""
Validate the conclusion.

Rules:
- No new information
- Logical completeness
- Citation correctness
""".strip(),
        expected_output="Validation report",
        agent=agent,
        context=[conclusion_task_ref],
    )


# ======================================================
# EDITING TASKS (NO RAG)
# ======================================================


def edit_intro_task(agent, intro_task_ref, validation_intro_task_ref):
    return Task(
        description="""
Improve academic style and clarity.

Rules:
- Preserve ALL citations
- Fix validation issues
- Do NOT add new content
STRICT RULE:
- Do NOT replace [source] with real document names
- Do NOT introduce SanPiN, laws, standards unless they already exist verbatim in the context
""".strip(),
        expected_output="Improved introduction",
        agent=agent,
        context=[intro_task_ref, validation_intro_task_ref],
        output_file="outputs/intro.txt",
    )


def edit_chap1_task(agent, chap1_task_ref, validation_chap1_task_ref):
    return Task(
        description="""
Improve academic style and clarity of Chapter 1.

Rules:
- Preserve ALL citations
- Fix validation issues
- Do NOT add new content
""".strip(),
        expected_output="Improved Chapter 1",
        agent=agent,
        context=[chap1_task_ref, validation_chap1_task_ref],
        output_file="outputs/chap1.txt",
    )


def edit_chap2_task(agent, chap2_task_ref, validation_chap2_task_ref):
    return Task(
        description="""
Improve academic style and clarity of Chapter 2.

Rules:
- Preserve ALL citations
- Fix validation issues
- Do NOT add new content
""".strip(),
        expected_output="Improved Chapter 2",
        agent=agent,
        context=[chap2_task_ref, validation_chap2_task_ref],
        output_file="outputs/chap2.txt",
    )


def edit_conclusion_task(agent, conclusion_task_ref, validation_conclusion_task_ref):
    return Task(
        description="""
Improve academic style and clarity of the conclusion.

Rules:
- Preserve ALL citations
- Fix validation issues
- Do NOT add new content
""".strip(),
        expected_output="Improved conclusion",
        agent=agent,
        context=[conclusion_task_ref, validation_conclusion_task_ref],
        output_file="outputs/conclusion.txt",
    )
