import logging

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from conversation_engine import ConversationEngine
from backend.provider import create_response_provider
from backend.schemas import ChatRequest, ChatResponse, HealthResponse

logger = logging.getLogger(__name__)

LOCAL_FRONTEND_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)

app = FastAPI(title="Therapy AI API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(LOCAL_FRONTEND_ORIGINS),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def get_engine():
    return ConversationEngine(create_response_provider())


@app.get("/api/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest, engine: ConversationEngine = Depends(get_engine)):
    try:
        result = engine.process_message(
            request.current_message,
            [message.model_dump() for message in request.messages],
            request.emotion_streak,
            request.stuck_state,
        )
    except Exception:
        logger.exception("Conversation processing failed.")
        raise HTTPException(
            status_code=503,
            detail="The conversation service is temporarily unavailable.",
        ) from None

    return ChatResponse(
        assistant_response=result.assistant_response,
        mood=result.mood,
        emotion_streak=result.emotion_streak,
        stuck_state=result.stuck_state,
        safety_category=result.safety_category,
    )
