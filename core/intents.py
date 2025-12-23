def detect_intent(message: str, state) -> str:
    msg = message.lower()

    if state.stage == "idle":
        return "start_coursework"

    if "план" in msg:
        return "planning"

    if "перепис" in msg or "исправ" in msg:
        return "revise"

    return "continue"
