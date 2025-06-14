# --- START OF FILE db.py ---

import sqlalchemy
from sqlalchemy import Table, Column, Integer, String, DateTime, LargeBinary, Text, JSON, Boolean, Index
from databases import Database
import os
from datetime import datetime, timezone
from dotenv import load_dotenv
import logging
from typing import Optional

logger = logging.getLogger(__name__)
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/emails.db")

database = Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()

emails = Table(
    "emails",
    metadata,
    Column("id", Integer, primary_key=True), # Auto-incrementing primary key
    Column("email_id", String, unique=True, index=True, nullable=False), # External email ID
    Column("account_id", String, index=True, nullable=False),
    Column("subject", String, nullable=True),
    Column("from_", String, nullable=True), # Renamed from 'from'
    Column("body", Text, nullable=True),
    Column("received_at", DateTime, index=True, nullable=False),
    Column("category", String, nullable=True),
    Column("embedding", LargeBinary, nullable=True),
    Column("full_text_for_embedding", Text, nullable=True), # Text used for embedding
    Column("is_unread", Boolean, default=True, nullable=True, index=True), # Added is_unread, default to True
)

chat_history = Table(
    "chat_history",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("conversation_id", String, index=True),
    Column("role", String),
    Column("content", Text),
    Column("timestamp", DateTime, default=lambda: datetime.now(timezone.utc)), # Ensure UTC now
)

# Ensure engine uses same connect_args if needed for specific DBs (like SQLite for check_same_thread)
engine = sqlalchemy.create_engine(DATABASE_URL)
if "sqlite" in DATABASE_URL:
    engine = sqlalchemy.create_engine(DATABASE_URL, connect_args={"check_same_thread": False})


async def migrate_database():
    """Add missing columns to existing database using ALTER TABLE for SQLite."""
    logger.info("Starting database migration check...")
    # This migration is basic, for complex changes Alembic is recommended
    try:
        async with database.transaction(): # Ensure atomic operations for migrations
            # Check and add 'from_' column
            # For SQLite, checking column existence is tricky. We'll try to add and ignore error if exists.
            # More robust check would query PRAGMA table_info('emails');
            try:
                await database.execute("ALTER TABLE emails ADD COLUMN from_ TEXT")
                logger.info("Added 'from_' column to emails table.")
            except Exception as e:
                if "duplicate column name" in str(e).lower() or "already exists" in str(e).lower():
                    logger.info("'from_' column already exists in emails table.")
                else:
                    logger.warning(f"Could not add 'from_' column (may exist or other error): {e}")

            # Check and add 'account_id' column and index
            try:
                await database.execute("ALTER TABLE emails ADD COLUMN account_id TEXT")
                logger.info("Added 'account_id' column to emails table.")
            except Exception as e:
                if "duplicate column name" in str(e).lower() or "already exists" in str(e).lower():
                    logger.info("'account_id' column already exists in emails table.")
                else:
                    logger.warning(f"Could not add 'account_id' column (may exist or other error): {e}")
            try:
                # For SQLite, ensure index creation is idempotent if possible, or handle error
                await database.execute("CREATE INDEX IF NOT EXISTS ix_emails_account_id ON emails (account_id)")
                logger.info("Ensured 'ix_emails_account_id' index exists on emails table for account_id.")
            except Exception as e:
                logger.warning(f"Could not create index for 'account_id': {e}")

            # Check and add 'is_unread' column and index
            try:
                await database.execute("ALTER TABLE emails ADD COLUMN is_unread BOOLEAN DEFAULT TRUE") # Default new to unread
                logger.info("Added 'is_unread' column to emails table (defaulting to TRUE).")
            except Exception as e:
                if "duplicate column name" in str(e).lower() or "already exists" in str(e).lower():
                    logger.info("'is_unread' column already exists in emails table.")
                else:
                    logger.warning(f"Could not add 'is_unread' column (may exist or other error): {e}")
            try:
                await database.execute("CREATE INDEX IF NOT EXISTS ix_emails_is_unread ON emails (is_unread)")
                logger.info("Ensured 'ix_emails_is_unread' index exists on emails table for is_unread.")
            except Exception as e:
                logger.warning(f"Could not create index for 'is_unread': {e}")
            
            # Check and add 'full_text_for_embedding'
            try:
                await database.execute("ALTER TABLE emails ADD COLUMN full_text_for_embedding TEXT")
                logger.info("Added 'full_text_for_embedding' column to emails table.")
            except Exception as e:
                if "duplicate column name" in str(e).lower() or "already exists" in str(e).lower():
                    logger.info("'full_text_for_embedding' column already exists in emails table.")
                else:
                    logger.warning(f"Could not add 'full_text_for_embedding' column (may exist or other error): {e}")


        logger.info("Database migration check completed.")
    except Exception as e:
        logger.error(f"Database migration failed: {e}", exc_info=True)

# Create tables if they don't exist (SQLAlchemy specific)
# This should run AFTER migrations for new columns if tables already exist.
# For new setups, it creates everything.
# It's often better to manage schema creation and migration separately (e.g. with Alembic)
# metadata.create_all(engine) # Moved to be called after connect in startup, or manage via Alembic


async def upsert_email(email_doc: dict):
    # email_doc comes from process_email, so 'received_at' should be ISO string
    # and 'embedding' should be bytes.
    
    # Convert received_at string to datetime object for DB
    received_at_val = email_doc.get('received_at')
    if isinstance(received_at_val, str):
        try:
            # Handle potential 'Z' and ensure it's parsed as UTC
            if received_at_val.endswith('Z'):
                dt_obj = datetime.fromisoformat(received_at_val.replace('Z', '+00:00'))
            else:
                dt_obj = datetime.fromisoformat(received_at_val)
            
            if dt_obj.tzinfo is None: # If parsed as naive
                dt_obj = dt_obj.replace(tzinfo=timezone.utc) # Assume UTC
            email_doc['received_at'] = dt_obj.astimezone(timezone.utc) # Standardize to UTC
        except ValueError:
            logger.error(f"Invalid ISO format for received_at: {received_at_val}. Using current UTC time.")
            email_doc['received_at'] = datetime.now(timezone.utc)
    elif not isinstance(received_at_val, datetime):
        logger.warning(f"received_at is not a string or datetime: {type(received_at_val)}. Using current UTC time.")
        email_doc['received_at'] = datetime.now(timezone.utc)
    elif received_at_val.tzinfo is None: # If it's a naive datetime object
        email_doc['received_at'] = received_at_val.replace(tzinfo=timezone.utc)
    else: # It's already an aware datetime object
        email_doc['received_at'] = received_at_val.astimezone(timezone.utc)


    # Ensure embedding is bytes
    embedding_val = email_doc.get('embedding')
    if embedding_val is not None and not isinstance(embedding_val, bytes):
        logger.warning(f"Embedding for {email_doc.get('email_id')} is not bytes, attempting conversion. Type: {type(embedding_val)}")
        # This case should ideally not happen if process_email returns bytes
        try:
            embedding_bytes = bytes(embedding_val) # Generic attempt
        except TypeError:
            logger.error(f"Could not convert embedding to bytes for {email_doc.get('email_id')}")
            embedding_bytes = None # Or handle error more strictly
    else:
        embedding_bytes = embedding_val

    values_to_insert_update = {
        "account_id": email_doc.get('account_id', 'unknown_account'),
        "subject": email_doc.get('subject', ''),
        "from_": email_doc.get('from_', ''),
        "body": email_doc.get('body', ''),
        "received_at": email_doc['received_at'], # Should be timezone-aware datetime object
        "category": email_doc.get('category', 'general'),
        "embedding": embedding_bytes,
        "full_text_for_embedding": email_doc.get('full_text_for_embedding', ''),
    }

    # Only add 'is_unread' to the values dict if it's explicitly provided (not None).
    # This allows the database/SQLAlchemy default to apply on INSERTs,
    # and prevents overwriting existing values with NULL on UPDATEs if not specified.
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
        # logger.debug(f"Updated email: {email_doc['email_id']}")
    else:
        insert_query = emails.insert().values(
            email_id=email_doc['email_id'],
            **values_to_insert_update
        )
        await database.execute(insert_query)
        # logger.debug(f"Inserted email: {email_doc['email_id']}")

async def get_email_by_id(email_id: str, account_id: Optional[str] = None):
    query = emails.select().where(emails.c.email_id == email_id)
    if account_id:
        query = query.where(emails.c.account_id == account_id)
    return await database.fetch_one(query)

async def get_emails_in_date_range(start_date: datetime, end_date: datetime, category: Optional[str] = None, account_id: Optional[str] = None, is_unread: Optional[bool] = None):
    # Ensure dates are timezone-aware UTC for comparison
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
    if is_unread is not None: # Allows filtering for True or False
        query = query.where(emails.c.is_unread == is_unread)

    query = query.order_by(emails.c.received_at.desc())
    return await database.fetch_all(query)

# get_emails_by_account_in_date_range is redundant if get_emails_in_date_range handles account_id

async def insert_chat_message(conversation_id: str, role: str, content: str): # Removed timestamp, use default
    query = chat_history.insert().values(
        conversation_id=conversation_id,
        role=role,
        content=content
        # timestamp will use default
    )
    await database.execute(query)

async def get_chat_history(conversation_id: str, limit: int = 20):
    query = (
        chat_history.select()
        .where(chat_history.c.conversation_id == conversation_id)
        .order_by(chat_history.c.timestamp.desc()) # Get latest first
        .limit(limit)
    )
    rows = await database.fetch_all(query)
    return list(reversed(rows))  # Reverse to get oldest first for chat display


async def update_email_status(email_id: str, is_unread: bool, account_id: str):
    query = (
        emails.update()
        .where(emails.c.email_id == email_id)
        .where(emails.c.account_id == account_id) # Important for security
        .values(is_unread=is_unread)
    )
    await database.execute(query)
    logger.info(f"Updated is_unread status for email {email_id} to {is_unread}")