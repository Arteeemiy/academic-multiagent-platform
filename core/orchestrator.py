from core.state import get_user_state
from core.intents import detect_intent
from core.crew_factory import run_coursework_crew
from core.execution_state import ExecutionState


class Orchestrator:

    async def handle(self, user_id: str, message: str):
        user_state = get_user_state(user_id)
        intent = detect_intent(message, user_state)

        if intent == "start_coursework":
            user_state.stage = "running"

            exec_state = ExecutionState(topic=message)

            result = await run_coursework_crew(
                topic=message,
                execution_state=exec_state,
            )

            user_state.stage = "done"

            return {
                "result": result,
                "pipeline_stage": exec_state.pipeline_stage,
                "timings": exec_state.timings,
                "artifacts": list(exec_state.artifacts.keys()),
            }
