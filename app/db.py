# --- START OF FILE db.py ---

import sqlalchemy
from sqlalchemy import Table, Column, Integer, String, DateTime, LargeBinary, Text, JSON, Boolean, Index
from databases import Database
import os
from datetime import datetime, timezone
from dotenv import load_dotenv
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/emails.db")

database = Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()

emails = Table(
    "emails",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("email_id", String, unique=True, index=True, nullable=False),
    Column("account_id", String, index=True, nullable=False),
    Column("subject", String, nullable=True),
    Column("from_", String, nullable=True),
    Column("body", Text, nullable=True),
    Column("received_at", DateTime, index=True, nullable=False),
    Column("category", String, nullable=True),
    Column("embedding", LargeBinary, nullable=True),
    Column("full_text_for_embedding", Text, nullable=True),
    Column("is_unread", Boolean, default=True, nullable=True, index=True),
)

chat_history = Table(
    "chat_history",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("conversation_id", String, index=True),
    Column("role", String),
    Column("content", Text),
    Column("timestamp", DateTime, default=lambda: datetime.now(timezone.utc)),
)

engine = sqlalchemy.create_engine(DATABASE_URL)
if "sqlite" in DATABASE_URL:
    engine = sqlalchemy.create_engine(DATABASE_URL, connect_args={"check_same_thread": False})


async def migrate_database():
    """Add missing columns to existing database using ALTER TABLE for SQLite."""
    logger.info("Starting database migration check...")
    try:
        async with database.transaction():
            try:
                await database.execute("ALTER TABLE emails ADD COLUMN from_ TEXT")
            except Exception: pass
            try:
                await database.execute("ALTER TABLE emails ADD COLUMN account_id TEXT")
            except Exception: pass
            try:
                await database.execute("CREATE INDEX IF NOT EXISTS ix_emails_account_id ON emails (account_id)")
            except Exception: pass
            try:
                await database.execute("ALTER TABLE emails ADD COLUMN is_unread BOOLEAN DEFAULT TRUE")
            except Exception: pass
            try:
                await database.execute("CREATE INDEX IF NOT EXISTS ix_emails_is_unread ON emails (is_unread)")
            except Exception: pass
            try:
                await database.execute("ALTER TABLE emails ADD COLUMN full_text_for_embedding TEXT")
            except Exception: pass
        logger.info("Database migration check completed.")
    except Exception as e:
        logger.error(f"Database migration failed: {e}", exc_info=True)


async def upsert_email(email_doc: dict):
    received_at_val = email_doc.get('received_at')
    if isinstance(received_at_val, str):
        try:
            if received_at_val.endswith('Z'):
                dt_obj = datetime.fromisoformat(received_at_val.replace('Z', '+00:00'))
            else:
                dt_obj = datetime.fromisoformat(received_at_val)
            if dt_obj.tzinfo is None:
                dt_obj = dt_obj.replace(tzinfo=timezone.utc)
            email_doc['received_at'] = dt_obj.astimezone(timezone.utc)
        except ValueError:
            email_doc['received_at'] = datetime.now(timezone.utc)
    elif not isinstance(received_at_val, datetime):
        email_doc['received_at'] = datetime.now(timezone.utc)
    elif received_at_val.tzinfo is None:
        email_doc['received_at'] = received_at_val.replace(tzinfo=timezone.utc)
    else:
        email_doc['received_at'] = received_at_val.astimezone(timezone.utc)

    embedding_val = email_doc.get('embedding')
    embedding_bytes = embedding_val if isinstance(embedding_val, bytes) else None
    values_to_insert_update = {
        "account_id": email_doc.get('account_id', 'unknown_account'),
        "subject": email_doc.get('subject', ''),
        "from_": email_doc.get('from_', ''),
        "body": email_doc.get('body', ''),
        "received_at": email_doc['received_at'],
        "category": email_doc.get('category', 'general'),
        "embedding": embedding_bytes,
        "full_text_for_embedding": email_doc.get('full_text_for_embedding', ''),
    }
    is_unread_status = email_doc.get('is_unread')
    if is_unread_status is not None:
        values_to_insert_update['is_unread'] = is_unread_status
    query = emails.select().where(emails.c.email_id == email_doc['email_id'])
    existing = await database.fetch_one(query)
    if existing:
        update_query = (
            emails.update()
            .where(emails.c.email_id == email_doc['email_id'])
            .values(**values_to_insert_update)
        )
        await database.execute(update_query)
    else:
        insert_query = emails.insert().values(
            email_id=email_doc['email_id'],
            **values_to_insert_update
        )
        await database.execute(insert_query)


async def get_email_by_id(email_id: str, account_id: Optional[str] = None):
    query = emails.select().where(emails.c.email_id == email_id)
    if account_id:
        query = query.where(emails.c.account_id == account_id)
    return await database.fetch_one(query)


async def get_emails_by_ids(email_ids: List[str], account_id: Optional[str] = None):
    """
    Fetches a list of emails given a list of email_ids, respecting account ownership.
    """
    if not email_ids:
        return []
    
    query = emails.select().where(emails.c.email_id.in_(email_ids))
    if account_id:
        query = query.where(emails.c.account_id == account_id)
    
    return await database.fetch_all(query)


async def get_emails_in_date_range(
    start_date: datetime,
    end_date: datetime,
    account_id: Optional[str] = None,
    is_unread: Optional[bool] = None,
    category: Optional[str] = None,
    limit: int = 50
):
    """
    Enhanced function to retrieve emails with multiple filter options, including category.
    """
    if start_date.tzinfo is None: start_date = start_date.replace(tzinfo=timezone.utc)
    else: start_date = start_date.astimezone(timezone.utc)
    if end_date.tzinfo is None: end_date = end_date.replace(tzinfo=timezone.utc)
    else: end_date = end_date.astimezone(timezone.utc)

    query = emails.select().where(
        (emails.c.received_at >= start_date) &
        (emails.c.received_at <= end_date)
    )
    if category:
        query = query.where(emails.c.category == category)
    if account_id:
        query = query.where(emails.c.account_id == account_id)
    if is_unread is not None:
        query = query.where(emails.c.is_unread == is_unread)

    query = query.order_by(emails.c.received_at.desc()).limit(limit)
    return await database.fetch_all(query)


async def insert_chat_message(conversation_id: str, role: str, content: str):
    query = chat_history.insert().values(
        conversation_id=conversation_id,
        role=role,
        content=content
    )
    await database.execute(query)


async def get_chat_history(conversation_id: str, limit: int = 20):
    query = (
        chat_history.select()
        .where(chat_history.c.conversation_id == conversation_id)
        .order_by(chat_history.c.timestamp.desc())
        .limit(limit)
    )
    rows = await database.fetch_all(query)
    return list(reversed(rows))


async def update_email_status(email_id: str, is_unread: bool, account_id: str):
    query = (
        emails.update()
        .where(emails.c.email_id == email_id)
        .where(emails.c.account_id == account_id)
        .values(is_unread=is_unread)
    )
    await database.execute(query)
    logger.info(f"Updated is_unread status for email {email_id} to {is_unread}")