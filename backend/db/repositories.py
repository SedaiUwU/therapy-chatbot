from __future__ import annotations

from collections.abc import Iterable
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .models import Conversation, Message, MessageRole


def create_conversation(session: Session, title: str | None = None) -> Conversation:
    """Add and flush a conversation without committing the caller's transaction."""
    conversation = Conversation(title=title)
    session.add(conversation)
    session.flush()
    return conversation


def get_conversation(session: Session, conversation_id: UUID) -> Conversation | None:
    return session.get(Conversation, conversation_id)


def delete_conversation(session: Session, conversation_id: UUID) -> bool:
    """Delete a conversation and flush; the caller owns commit or rollback."""
    conversation = get_conversation(session, conversation_id)
    if conversation is None:
        return False

    session.delete(conversation)
    session.flush()
    return True


def update_conversation_state(
    session: Session,
    conversation_id: UUID,
    emotion_streak: int,
    stuck_state: int,
) -> Conversation | None:
    conversation = get_conversation(session, conversation_id)
    if conversation is None:
        return None

    conversation.emotion_streak = emotion_streak
    conversation.stuck_state = stuck_state
    session.flush()
    return conversation


def append_message(
    session: Session,
    conversation_id: UUID,
    role: MessageRole | str,
    content: str,
) -> Message:
    """Append under a parent-row lock; flush but leave transaction ownership to caller."""
    try:
        message_role = role if isinstance(role, MessageRole) else MessageRole(role)
    except ValueError as exc:
        raise ValueError("role must be 'user' or 'assistant'") from exc

    conversation = session.scalar(
        select(Conversation)
        .where(Conversation.id == conversation_id)
        .with_for_update()
    )
    if conversation is None:
        raise LookupError("conversation does not exist")

    last_sequence = session.scalar(
        select(func.max(Message.sequence)).where(
            Message.conversation_id == conversation_id
        )
    )
    next_sequence = 0 if last_sequence is None else last_sequence + 1

    conversation.updated_at = func.current_timestamp()
    message = Message(
        conversation_id=conversation_id,
        role=message_role,
        content=content,
        sequence=next_sequence,
    )
    session.add(message)
    session.flush()
    return message


def get_messages_for_conversation(
    session: Session,
    conversation_id: UUID,
) -> list[Message] | None:
    """Return ordered messages, [] for an existing empty conversation, or None if missing."""
    if get_conversation(session, conversation_id) is None:
        return None

    return list(
        session.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.sequence)
        ).all()
    )


def messages_to_history(messages: Iterable[Message]) -> list[dict[str, str]]:
    """Convert persisted messages into the plain history consumed by ConversationEngine."""
    return [
        {"role": message.role.value, "content": message.content}
        for message in messages
    ]
