"""create conversations and messages

Revision ID: 20260927_000001
Revises:
Create Date: 2026-09-27 00:00:01.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "20260927_000001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "conversations",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("emotion_streak", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("stuck_state", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("title", sa.String(length=255), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_conversations_created_at"), "conversations", ["created_at"], unique=False)
    op.create_index(op.f("ix_conversations_updated_at"), "conversations", ["updated_at"], unique=False)
    op.create_table(
        "messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("role", sa.Enum("user", "assistant", name="message_role"), nullable=False),
        sa.Column("content", sa.String(length=10000), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("conversation_id", "sequence", name="uq_messages_conversation_sequence"),
    )


def downgrade() -> None:
    op.drop_table("messages")
    op.drop_index(op.f("ix_conversations_updated_at"), table_name="conversations")
    op.drop_index(op.f("ix_conversations_created_at"), table_name="conversations")
    op.drop_table("conversations")
    op.execute("DROP TYPE IF EXISTS message_role")
