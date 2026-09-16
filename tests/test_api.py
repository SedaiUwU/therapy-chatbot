from dataclasses import dataclass

from fastapi.testclient import TestClient

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
    assert "ChatRequest" in schema["components"]["schemas"]
    assert "ChatResponse" in schema["components"]["schemas"]
    assert "GROQ_API_KEY" not in serialized
    assert "api_key" not in serialized
    assert "provider" not in serialized
    assert "prompt" not in serialized
