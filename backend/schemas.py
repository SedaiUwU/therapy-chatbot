from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator


MessageRole = Literal["user", "assistant"]


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: MessageRole
    content: str = Field(min_length=1, max_length=10_000)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value):
        if not value.strip():
            raise ValueError("content must not be blank")
        return value


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_message: str = Field(min_length=1, max_length=10_000)
    messages: list[Message] = Field(default_factory=list, max_length=100)
    emotion_streak: StrictInt = Field(default=0, ge=0, le=1_000)
    stuck_state: StrictInt = Field(default=0, ge=0, le=1_000)

    @field_validator("current_message")
    @classmethod
    def current_message_must_not_be_blank(cls, value):
        if not value.strip():
            raise ValueError("current_message must not be blank")
        return value


class ChatResponse(BaseModel):
    assistant_response: str
    mood: str | None
    emotion_streak: int
    stuck_state: int
    safety_category: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
