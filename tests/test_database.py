from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from sqlalchemy import Uuid, create_engine, event, func, select
from sqlalchemy.exc import StatementError
from sqlalchemy.orm import Session

from backend.db.base import Base
from backend.db.models import Conversation, Message, MessageRole
from backend.db.repositories import (
    append_message,
    create_conversation,
    delete_conversation,
    get_conversation,
    get_messages_for_conversation,
    messages_to_history,
    update_conversation_state,
)


def build_session():
    engine = create_engine("sqlite:///:memory:", future=True)

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, _record):
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=engine)
    return engine, Session(engine)


def test_conversation_can_be_created_with_default_state():
    engine, session = build_session()
    try:
        conversation = Conversation()
        session.add(conversation)
        session.commit()
        session.refresh(conversation)

        assert conversation.emotion_streak == 0
        assert conversation.stuck_state == 0
        assert conversation.id is not None
        assert conversation.created_at is not None
        assert conversation.updated_at is not None
    finally:
        session.close()
        engine.dispose()


def test_orm_metadata_matches_persistence_contract():
    conversation_columns = Conversation.__table__.c
    message_columns = Message.__table__.c

    assert isinstance(conversation_columns.id.type, Uuid)
    assert isinstance(message_columns.id.type, Uuid)
    assert isinstance(message_columns.conversation_id.type, Uuid)
    assert conversation_columns.created_at.type.timezone is True
    assert conversation_columns.updated_at.type.timezone is True
    assert message_columns.created_at.type.timezone is True
    assert conversation_columns.created_at.server_default is not None
    assert conversation_columns.updated_at.server_default is not None
    assert message_columns.created_at.server_default is not None
    assert str(conversation_columns.emotion_streak.server_default.arg) == "0"
    assert str(conversation_columns.stuck_state.server_default.arg) == "0"
    assert {index.name for index in Conversation.__table__.indexes} == {
        "ix_conversations_created_at",
        "ix_conversations_updated_at",
    }
    assert Message.__table__.indexes == set()
    assert {
        (constraint.name, tuple(column.name for column in constraint.columns))
        for constraint in Message.__table__.constraints
        if constraint.name == "uq_messages_conversation_sequence"
    } == {
        ("uq_messages_conversation_sequence", ("conversation_id", "sequence"))
    }
    assert Message.__table__.c.role.type.enums == ["user", "assistant"]


def test_message_belongs_to_conversation_and_role_is_valid():
    engine, session = build_session()
    try:
        conversation = Conversation()
        session.add(conversation)
        session.commit()
        session.refresh(conversation)

        message = Message(
            conversation_id=conversation.id,
            role=MessageRole.USER,
            content="I am feeling anxious.",
            sequence=0,
        )
        session.add(message)
        session.commit()
        session.refresh(message)

        assert message.conversation_id == conversation.id
        assert message.role == MessageRole.USER
        assert message.content == "I am feeling anxious."
        assert message.sequence == 0
    finally:
        session.close()
        engine.dispose()


def test_message_ordering_is_deterministic_with_sequence():
    engine, session = build_session()
    try:
        conversation = Conversation()
        session.add(conversation)
        session.commit()
        session.refresh(conversation)

        session.add_all(
            [
                Message(
                    conversation_id=conversation.id,
                    role=MessageRole.USER,
                    content="First",
                    sequence=1,
                ),
                Message(
                    conversation_id=conversation.id,
                    role=MessageRole.ASSISTANT,
                    content="Second",
                    sequence=2,
                ),
                Message(
                    conversation_id=conversation.id,
                    role=MessageRole.USER,
                    content="Third",
                    sequence=3,
                ),
            ]
        )
        session.commit()

        ordered = session.scalars(
            select(Message)
            .where(Message.conversation_id == conversation.id)
            .order_by(Message.sequence)
        ).all()

        assert [item.content for item in ordered] == ["First", "Second", "Third"]
    finally:
        session.close()
        engine.dispose()


def test_conversation_message_relationship_and_cascade_are_configured():
    engine, session = build_session()
    try:
        conversation = Conversation()
        session.add(conversation)
        session.commit()
        session.refresh(conversation)

        session.add_all(
            [
                Message(
                    conversation_id=conversation.id,
                    role=MessageRole.USER,
                    content="Hello",
                    sequence=0,
                ),
                Message(
                    conversation_id=conversation.id,
                    role=MessageRole.ASSISTANT,
                    content="Hi there",
                    sequence=1,
                ),
            ]
        )
        session.commit()

        assert len(conversation.messages) == 2
        session.delete(conversation)
        session.commit()

        assert session.query(Message).count() == 0
    finally:
        session.close()
        engine.dispose()


def test_message_role_constraint_and_schema_exclude_hidden_fields():
    engine, session = build_session()
    try:
        conversation = Conversation()
        session.add(conversation)
        session.commit()
        session.refresh(conversation)

        invalid_message = Message(
            conversation_id=conversation.id,
            role="system",
            content="Not allowed",
            sequence=0,
        )
        session.add(invalid_message)

        try:
            session.commit()
        except StatementError:
            session.rollback()
        else:
            assert False, "system role should be rejected by schema validation"

        persisted_columns = set(Conversation.__table__.columns.keys()) | set(Message.__table__.columns.keys())
        assert {"api_key", "provider", "prompt", "model", "secret"}.isdisjoint(persisted_columns)
    finally:
        session.close()
        engine.dispose()


def test_repository_creates_and_retrieves_conversation():
    engine, session = build_session()
    try:
        conversation = create_conversation(session, title="A short title")
        conversation_id = conversation.id
        session.commit()

        assert isinstance(conversation_id, UUID)
        assert conversation.emotion_streak == 0
        assert conversation.stuck_state == 0
        assert conversation.title == "A short title"
        assert get_conversation(session, conversation_id) is conversation
        assert get_conversation(session, uuid4()) is None
    finally:
        session.close()
        engine.dispose()


def test_repository_state_update_only_changes_state_and_updates_timestamp():
    engine, session = build_session()
    try:
        conversation = create_conversation(session, title="Keep this title")
        conversation.created_at = datetime(2000, 1, 1, tzinfo=timezone.utc)
        session.flush()
        conversation_id = conversation.id

        updated = update_conversation_state(session, conversation_id, 3, 2)
        session.commit()

        assert updated is conversation
        assert conversation.emotion_streak == 3
        assert conversation.stuck_state == 2
        assert conversation.title == "Keep this title"
        assert conversation.created_at.year == 2000
        assert conversation.updated_at.year > 2000
        assert update_conversation_state(session, uuid4(), 1, 1) is None
    finally:
        session.close()
        engine.dispose()


def test_repository_appends_and_retrieves_ordered_plain_history():
    engine, session = build_session()
    try:
        conversation = create_conversation(session)
        first = append_message(
            session,
            conversation.id,
            "user",
            "Repository test user text",
        )
        second = append_message(
            session,
            conversation.id,
            MessageRole.ASSISTANT,
            "Repository test assistant text",
        )
        session.commit()

        assert (first.sequence, second.sequence) == (0, 1)
        messages = get_messages_for_conversation(session, conversation.id)
        assert messages is not None
        assert messages_to_history(messages) == [
            {"role": "user", "content": "Repository test user text"},
            {"role": "assistant", "content": "Repository test assistant text"},
        ]
        assert get_messages_for_conversation(session, uuid4()) is None
    finally:
        session.close()
        engine.dispose()


def test_repository_empty_history_and_invalid_role_behavior():
    engine, session = build_session()
    try:
        conversation = create_conversation(session)
        assert get_messages_for_conversation(session, conversation.id) == []

        with pytest.raises(ValueError, match="role must be 'user' or 'assistant'"):
            append_message(session, conversation.id, "system", "Not persisted")

        assert session.scalar(select(func.count()).select_from(Message)) == 0
    finally:
        session.close()
        engine.dispose()


def test_repository_append_requires_existing_conversation():
    engine, session = build_session()
    try:
        with pytest.raises(LookupError, match="conversation does not exist"):
            append_message(session, uuid4(), "user", "No orphan")

        assert session.scalar(select(func.count()).select_from(Conversation)) == 0
        assert session.scalar(select(func.count()).select_from(Message)) == 0
    finally:
        session.close()
        engine.dispose()


def test_repository_delete_cascades_messages():
    engine, session = build_session()
    try:
        conversation = create_conversation(session)
        append_message(session, conversation.id, "user", "Delete with parent")
        conversation_id = conversation.id
        session.commit()

        assert delete_conversation(session, conversation_id) is True
        session.commit()

        assert get_conversation(session, conversation_id) is None
        assert session.scalar(select(func.count()).select_from(Message)) == 0
        assert delete_conversation(session, conversation_id) is False
    finally:
        session.close()
        engine.dispose()


def test_repository_transaction_rollback_removes_prior_writes():
    engine, session = build_session()
    try:
        conversation = create_conversation(session)
        append_message(session, conversation.id, "user", "Will roll back")

        with pytest.raises(ValueError):
            append_message(session, conversation.id, "system", "Rejected role")

        session.rollback()
        assert session.scalar(select(func.count()).select_from(Conversation)) == 0
        assert session.scalar(select(func.count()).select_from(Message)) == 0
    finally:
        session.close()
        engine.dispose()


def test_conversation_engine_imports_without_sqlalchemy_available():
    isolated_import = """
import builtins

original_import = builtins.__import__
def import_without_sqlalchemy(name, *args, **kwargs):
    if name == 'sqlalchemy' or name.startswith('sqlalchemy.'):
        raise AssertionError('ConversationEngine imported SQLAlchemy')
    return original_import(name, *args, **kwargs)

builtins.__import__ = import_without_sqlalchemy
from conversation_engine import ConversationEngine

engine = ConversationEngine(lambda prompt: 'A deterministic reply.')
result = engine.process_message('Hello', [])
assert result.assistant_response == 'A deterministic reply.'
"""
    subprocess.run([sys.executable, "-c", isolated_import], check=True)
