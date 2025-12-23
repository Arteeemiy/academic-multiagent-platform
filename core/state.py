from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class UserState:
    stage: str = "idle"
    topic: str | None = None
    history: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


_STATE: dict[str, UserState] = {}


def get_user_state(user_id: str) -> UserState:
    if user_id not in _STATE:
        _STATE[user_id] = UserState()
    return _STATE[user_id]
