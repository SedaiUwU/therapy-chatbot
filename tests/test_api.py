from dataclasses import dataclass
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import backend.main as main_module
from backend.db.base import Base
from backend.db.models import Message
from backend.db.session import get_db_session
from backend.main import app, get_engine


@dataclass
class FakeResult:
    assistant_response: str = "That sounds difficult."
    mood: str | None = "stressed"
    emotion_streak: int = 2
    stuck_state: int = 0
    safety_category: str | None = None


class FakeEngine:
    def __init__(self):
        self.calls = []

    def process_message(self, user_input, messages, emotion_streak=0, stuck_state=0):
        self.calls.append((user_input, messages, emotion_streak, stuck_state))
        return FakeResult()


class PersistedFakeEngine:
    def __init__(self):
        self.calls = []

    def process_message(self, user_input, messages, emotion_streak=0, stuck_state=0):
        self.calls.append((user_input, messages, emotion_streak, stuck_state))
        return FakeResult(
            assistant_response=f"Persisted reply {len(self.calls)}",
            mood="neutral",
            emotion_streak=emotion_streak + 1,
            stuck_state=stuck_state + 1,
        )


@pytest.fixture
def persistent_client():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, _record):
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    fake_engine = PersistedFakeEngine()

    def override_db_session():
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_db_session
    app.dependency_overrides[get_engine] = lambda: fake_engine
    try:
        with TestClient(app) as client:
            yield client, fake_engine, session_factory
    finally:
        app.dependency_overrides.pop(get_db_session, None)
        app.dependency_overrides.pop(get_engine, None)
        engine.dispose()


def test_health_returns_public_status_contract():
    response = TestClient(app).get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_validates_request_and_maps_engine_result_without_provider_call():
    engine = FakeEngine()
    app.dependency_overrides[get_engine] = lambda: engine
    try:
        response = TestClient(app).post(
            "/api/chat",
            json={
                "current_message": "I feel stressed.",
                "messages": [
                    {"role": "user", "content": "My exam is tomorrow."},
                    {"role": "assistant", "content": "That is coming up soon."},
                ],
                "emotion_streak": 1,
                "stuck_state": 0,
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "assistant_response": "That sounds difficult.",
        "mood": "stressed",
        "emotion_streak": 2,
        "stuck_state": 0,
        "safety_category": None,
    }
    assert engine.calls == [
        (
            "I feel stressed.",
            [
                {"role": "user", "content": "My exam is tomorrow."},
                {"role": "assistant", "content": "That is coming up soon."},
            ],
            1,
            0,
        )
    ]


def test_chat_supports_two_turn_client_managed_history_and_state():
    class RoundTripEngine:
        def __init__(self):
            self.calls = []

        def process_message(self, user_input, messages, emotion_streak=0, stuck_state=0):
            self.calls.append((user_input, messages, emotion_streak, stuck_state))
            return FakeResult(
                assistant_response=f"Reply to {user_input}",
                mood="neutral",
                emotion_streak=emotion_streak + 1,
                stuck_state=stuck_state + 1,
            )

    engine = RoundTripEngine()
    client = TestClient(app)
    app.dependency_overrides[get_engine] = lambda: engine
    try:
        first = client.post(
            "/api/chat",
            json={"current_message": "Hello"},
        )
        first_body = first.json()
        second = client.post(
            "/api/chat",
            json={
                "current_message": "I am still thinking about it.",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": first_body["assistant_response"]},
                ],
                "emotion_streak": first_body["emotion_streak"],
                "stuck_state": first_body["stuck_state"],
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["emotion_streak"] == 2
    assert second.json()["stuck_state"] == 2
    assert engine.calls == [
        ("Hello", [], 0, 0),
        (
            "I am still thinking about it.",
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Reply to Hello"},
            ],
            1,
            1,
        ),
    ]


def test_chat_rejects_invalid_message_role_and_missing_current_message():
    invalid_role = TestClient(app).post(
        "/api/chat",
        json={
            "current_message": "Hello",
            "messages": [{"role": "system", "content": "Hidden prompt"}],
        },
    )
    missing_message = TestClient(app).post("/api/chat", json={"messages": []})

    assert invalid_role.status_code == 422
    assert missing_message.status_code == 422


def test_chat_rejects_blank_content_unknown_fields_and_non_integer_state():
    response = TestClient(app).post(
        "/api/chat",
        json={
            "current_message": "   ",
            "messages": [{"role": "user", "content": "Hello", "extra": "value"}],
            "emotion_streak": True,
            "unexpected": "value",
        },
    )

    assert response.status_code == 422


def test_chat_hides_internal_engine_failure():
    class BrokenEngine:
        def process_message(self, *args, **kwargs):
            raise RuntimeError("provider secret and prompt details")

    app.dependency_overrides[get_engine] = lambda: BrokenEngine()
    try:
        response = TestClient(app).post(
            "/api/chat",
            json={"current_message": "Hello"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {
        "detail": "The conversation service is temporarily unavailable."
    }


def test_cors_allows_frontend_origins_and_rejects_unrelated_origin():
    client = TestClient(app)
    for origin in ("http://localhost:5173", "http://127.0.0.1:5173"):
        response = client.options(
            "/api/chat",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == origin

    delete_preflight = client.options(
        "/api/conversations/00000000-0000-0000-0000-000000000001",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "DELETE",
        },
    )
    assert delete_preflight.status_code == 200
    assert "DELETE" in delete_preflight.headers["access-control-allow-methods"]

    unrelated = client.options(
        "/api/chat",
        headers={
            "Origin": "https://unrelated.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert unrelated.status_code == 400
    assert "access-control-allow-origin" not in unrelated.headers


def test_openapi_documents_public_routes_and_schemas_without_provider_details():
    schema = TestClient(app).get("/openapi.json").json()
    serialized = str(schema)

    assert "/api/health" in schema["paths"]
    assert "/api/chat" in schema["paths"]
    assert "/api/conversations" in schema["paths"]
    assert "/api/conversations/{conversation_id}" in schema["paths"]
    assert "/api/conversations/{conversation_id}/chat" in schema["paths"]
    assert "ChatRequest" in schema["components"]["schemas"]
    assert "ChatResponse" in schema["components"]["schemas"]
    assert "ConversationCreateRequest" in schema["components"]["schemas"]
    assert "ConversationHistoryResponse" in schema["components"]["schemas"]
    assert "PersistedChatRequest" in schema["components"]["schemas"]
    assert "PersistedChatResponse" in schema["components"]["schemas"]
    assert "MessageResponse" in schema["components"]["schemas"]
    assert schema["components"]["schemas"]["PersistedChatRequest"]["properties"].keys() == {
        "current_message"
    }
    assert "DATABASE_URL" not in serialized
    assert "GROQ_API_KEY" not in serialized
    assert "api_key" not in serialized
    assert "provider" not in serialized
    assert "prompt" not in serialized


def test_persistent_conversation_create_and_empty_retrieval(persistent_client):
    client, _engine, _session_factory = persistent_client

    created = client.post("/api/conversations", json={"title": "API test"})
    body = created.json()

    assert created.status_code == 201
    assert isinstance(UUID(body["id"]), UUID)
    assert body["title"] == "API test"
    assert (body["emotion_streak"], body["stuck_state"]) == (0, 0)

    retrieved = client.get(f"/api/conversations/{body['id']}")
    assert retrieved.status_code == 200
    assert retrieved.json()["messages"] == []


def test_persistent_missing_conversation_routes_return_404(persistent_client):
    client, engine, _session_factory = persistent_client
    missing_id = "00000000-0000-0000-0000-000000000123"

    assert client.get(f"/api/conversations/{missing_id}").status_code == 404
    assert client.post(
        f"/api/conversations/{missing_id}/chat",
        json={"current_message": "Synthetic request"},
    ).status_code == 404
    assert client.delete(f"/api/conversations/{missing_id}").status_code == 404
    assert engine.calls == []


def test_persisted_first_and_second_turn_round_trip_history_and_state(persistent_client):
    client, engine, _session_factory = persistent_client
    created = client.post("/api/conversations", json={}).json()
    conversation_id = created["id"]

    first = client.post(
        f"/api/conversations/{conversation_id}/chat",
        json={"current_message": "First synthetic turn"},
    )
    second = client.post(
        f"/api/conversations/{conversation_id}/chat",
        json={"current_message": "Second synthetic turn"},
    )

    assert first.status_code == 200
    assert first.json()["conversation_id"] == conversation_id
    assert first.json()["emotion_streak"] == 1
    assert first.json()["stuck_state"] == 1
    assert second.status_code == 200
    assert second.json()["emotion_streak"] == 2
    assert second.json()["stuck_state"] == 2
    assert engine.calls == [
        ("First synthetic turn", [], 0, 0),
        (
            "Second synthetic turn",
            [
                {"role": "user", "content": "First synthetic turn"},
                {"role": "assistant", "content": "Persisted reply 1"},
            ],
            1,
            1,
        ),
    ]

    history = client.get(f"/api/conversations/{conversation_id}").json()
    assert (history["emotion_streak"], history["stuck_state"]) == (2, 2)
    assert [message["sequence"] for message in history["messages"]] == [0, 1, 2, 3]
    assert [(message["role"], message["content"]) for message in history["messages"]] == [
        ("user", "First synthetic turn"),
        ("assistant", "Persisted reply 1"),
        ("user", "Second synthetic turn"),
        ("assistant", "Persisted reply 2"),
    ]
    assert "Second synthetic turn" not in [
        message["content"] for message in engine.calls[1][1]
    ]


def test_persistent_delete_removes_conversation_and_messages(persistent_client):
    client, _engine, session_factory = persistent_client
    conversation_id = client.post("/api/conversations", json={}).json()["id"]
    client.post(
        f"/api/conversations/{conversation_id}/chat",
        json={"current_message": "Delete cascade test"},
    )

    deleted = client.delete(f"/api/conversations/{conversation_id}")

    assert deleted.status_code == 204
    assert client.get(f"/api/conversations/{conversation_id}").status_code == 404
    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(Message)) == 0


def test_persisted_turn_failure_rolls_back_messages_and_state(
    persistent_client,
    monkeypatch,
):
    client, _engine, _session_factory = persistent_client
    conversation_id = client.post("/api/conversations", json={}).json()["id"]
    original_update = main_module.update_conversation_state

    def update_then_fail(*args, **kwargs):
        original_update(*args, **kwargs)
        raise RuntimeError("private database detail")

    monkeypatch.setattr(main_module, "update_conversation_state", update_then_fail)
    response = client.post(
        f"/api/conversations/{conversation_id}/chat",
        json={"current_message": "Rollback this synthetic turn"},
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "The conversation service is temporarily unavailable."
    }
    history = client.get(f"/api/conversations/{conversation_id}").json()
    assert (history["emotion_streak"], history["stuck_state"]) == (0, 0)
    assert history["messages"] == []


def test_stateless_chat_does_not_require_database(persistent_client):
    client, _engine, _session_factory = persistent_client

    def fail_if_database_is_requested():
        pytest.fail("stateless /api/chat must not request a database session")

    app.dependency_overrides[get_db_session] = fail_if_database_is_requested
    response = client.post(
        "/api/chat",
        json={"current_message": "Still stateless"},
    )

    assert response.status_code == 200
