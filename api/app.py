from fastapi import FastAPI
from pydantic import BaseModel

from core.orchestrator import Orchestrator

app = FastAPI(title="Academic Multi-Agent System")

orchestrator = Orchestrator()


class ChatRequest(BaseModel):
    user_id: str
    message: str


@app.post("/chat")
async def chat(req: ChatRequest):
    response = await orchestrator.handle(
        user_id=req.user_id,
        message=req.message,
    )

    return response
