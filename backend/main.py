import logging
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.orm import Session

from conversation_engine import ConversationEngine
from backend.provider import create_response_provider
from backend.db.models import Conversation
from backend.db.repositories import (
    append_message,
    create_conversation as repository_create_conversation,
    delete_conversation as repository_delete_conversation,
    get_conversation as repository_get_conversation,
    get_messages_for_conversation,
    messages_to_history,
    update_conversation_state,
)
from backend.db.session import get_db_session
from backend.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationCreateRequest,
    ConversationHistoryResponse,
    ConversationResponse,
    HealthResponse,
    MessageResponse,
    PersistedChatRequest,
    PersistedChatResponse,
)

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
    allow_methods=["GET", "POST", "DELETE"],
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


def _conversation_response(conversation: Conversation) -> ConversationResponse:
    return ConversationResponse(
        id=conversation.id,
        title=conversation.title,
        emotion_streak=conversation.emotion_streak,
        stuck_state=conversation.stuck_state,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


def _raise_persistence_error() -> None:
    logger.exception("Persistent conversation operation failed.")
    raise HTTPException(
        status_code=503,
        detail="The conversation service is temporarily unavailable.",
    ) from None


@app.post(
    "/api/conversations",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_persistent_conversation(
    request: ConversationCreateRequest,
    session: Session = Depends(get_db_session),
):
    try:
        with session.begin():
            conversation = repository_create_conversation(session, request.title)
            response = _conversation_response(conversation)
        return response
    except Exception:
        _raise_persistence_error()


@app.get(
    "/api/conversations/{conversation_id}",
    response_model=ConversationHistoryResponse,
)
def get_persistent_conversation(
    conversation_id: UUID,
    session: Session = Depends(get_db_session),
):
    conversation = repository_get_conversation(session, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    messages = get_messages_for_conversation(session, conversation_id)
    return ConversationHistoryResponse(
        **_conversation_response(conversation).model_dump(),
        messages=[
            MessageResponse(
                id=message.id,
                role=message.role.value,
                content=message.content,
                sequence=message.sequence,
                created_at=message.created_at,
            )
            for message in messages or []
        ],
    )


@app.delete(
    "/api/conversations/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
def delete_persistent_conversation(
    conversation_id: UUID,
    session: Session = Depends(get_db_session),
):
    try:
        with session.begin():
            deleted = repository_delete_conversation(session, conversation_id)
            if not deleted:
                raise HTTPException(status_code=404, detail="Conversation not found.")
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except HTTPException:
        raise
    except Exception:
        _raise_persistence_error()


@app.post(
    "/api/conversations/{conversation_id}/chat",
    response_model=PersistedChatResponse,
)
def persisted_chat(
    conversation_id: UUID,
    request: PersistedChatRequest,
    engine: ConversationEngine = Depends(get_engine),
    session: Session = Depends(get_db_session),
):
    try:
        # This serializes stored turns but holds the row lock through model latency.
        with session.begin():
            conversation = session.scalar(
                select(Conversation)
                .where(Conversation.id == conversation_id)
                .with_for_update()
            )
            if conversation is None:
                raise HTTPException(status_code=404, detail="Conversation not found.")

            stored_messages = get_messages_for_conversation(session, conversation_id)
            prior_history = messages_to_history(stored_messages or [])
            result = engine.process_message(
                user_input=request.current_message,
                messages=prior_history,
                emotion_streak=conversation.emotion_streak,
                stuck_state=conversation.stuck_state,
            )

            append_message(session, conversation_id, "user", request.current_message)
            append_message(
                session,
                conversation_id,
                "assistant",
                result.assistant_response,
            )
            update_conversation_state(
                session,
                conversation_id,
                result.emotion_streak,
                result.stuck_state,
            )
            response = PersistedChatResponse(
                conversation_id=conversation_id,
                assistant_response=result.assistant_response,
                mood=result.mood,
                emotion_streak=result.emotion_streak,
                stuck_state=result.stuck_state,
                safety_category=result.safety_category,
            )
        return response
    except HTTPException:
        raise
    except Exception:
        _raise_persistence_error()
