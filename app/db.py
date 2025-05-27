import sqlalchemy
from sqlalchemy import Table, Column, Integer, String, DateTime, LargeBinary, Text, JSON
from databases import Database
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/emails.db")

database = Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()

emails = Table(
    "emails",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("email_id", String, unique=True, index=True),
    Column("account_id", String, index=True),  # Added account_id column
    Column("subject", String),
    Column("from_", String),
    Column("body", Text),
    Column("received_at", DateTime),
    Column("category", String),
    Column("embedding", LargeBinary),
    Column("full_text", Text),
)

chat_history = Table(
    "chat_history",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("conversation_id", String, index=True),
    Column("role", String),
    Column("content", Text),
    Column("timestamp", DateTime, default=datetime.utcnow),
)

engine = sqlalchemy.create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30}
)

# Create tables - this will add missing columns
metadata.create_all(engine)

async def migrate_database():
    """Add missing columns to existing database"""
    try:
        async with database.transaction():
            # Check for 'from_' column
            try:
                await database.fetch_one("SELECT from_ FROM emails LIMIT 1")
                print("'from_' column already exists in emails table.")
            except Exception:
                print("Adding 'from_' column to emails table.")
                await database.execute("ALTER TABLE emails ADD COLUMN from_ TEXT")

            # Check for 'account_id' column
            try:
                await database.fetch_one("SELECT account_id FROM emails LIMIT 1")
                print("'account_id' column already exists in emails table.")
            except Exception:
                print("Adding 'account_id' column to emails table.")
                await database.execute("ALTER TABLE emails ADD COLUMN account_id TEXT")
                # You might want to add an index to account_id for better query performance
                await database.execute("CREATE INDEX IF NOT EXISTS ix_emails_account_id ON emails (account_id)")

    except Exception as e:
        print(f"Migration error: {e}")

# DB helper functions
async def upsert_email(email_doc):
    # Convert received_at string to datetime if needed
    if isinstance(email_doc.get('received_at'), str):
        email_doc['received_at'] = datetime.fromisoformat(email_doc['received_at'])
        
    embedding_bytes = bytes(email_doc['embedding']) if not isinstance(email_doc['embedding'], bytes) else email_doc['embedding']

    query = emails.select().where(emails.c.email_id == email_doc['email_id'])
    existing = await database.fetch_one(query)
    if existing:
        update_query = (
            emails.update()
            .where(emails.c.email_id == email_doc['email_id'])
            .values(
                account_id=email_doc.get('account_id', 'unknown_account'), # Store account_id
                subject=email_doc.get('subject', ''),
                from_=email_doc.get('from_', ''),
                body=email_doc.get('body', ''),
                received_at=email_doc['received_at'],
                category=email_doc.get('category', 'general'),
                embedding=embedding_bytes,
                full_text=email_doc.get('full_text', '')
            )
        )
        await database.execute(update_query)
    else:
        insert_query = emails.insert().values(
            email_id=email_doc['email_id'],
            account_id=email_doc.get('account_id', 'unknown_account'), # Store account_id
            subject=email_doc.get('subject', ''),
            from_=email_doc.get('from_', ''),
            body=email_doc.get('body', ''),
            received_at=email_doc['received_at'],
            category=email_doc.get('category', 'general'),
            embedding=embedding_bytes,
            full_text=email_doc.get('full_text', '')
        )
        await database.execute(insert_query)

async def get_email_by_id(email_id: str, account_id: str = None):
    query = emails.select().where(emails.c.email_id == email_id)
    if account_id:
        query = query.where(emails.c.account_id == account_id)
    return await database.fetch_one(query)

async def get_emails_in_date_range(start_date, end_date, category=None, account_id: str = None):
    query = emails.select().where(
        (emails.c.received_at >= start_date) &
        (emails.c.received_at <= end_date)
    )
    if category:
        query = query.where(emails.c.category == category)
    if account_id:
        query = query.where(emails.c.account_id == account_id)
    query = query.order_by(emails.c.received_at.desc())
    return await database.fetch_all(query)

async def get_emails_by_account_in_date_range(account_id: str, start_date, end_date, category=None):
    """Fetches emails for a specific account within a date range."""
    query = emails.select().where(
        (emails.c.account_id == account_id) &
        (emails.c.received_at >= start_date) &
        (emails.c.received_at <= end_date)
    )
    if category:
        query = query.where(emails.c.category == category)
    query = query.order_by(emails.c.received_at.desc())
    return await database.fetch_all(query)


async def insert_chat_message(conversation_id, role, content, timestamp):
    query = chat_history.insert().values(
        conversation_id=conversation_id,
        role=role,
        content=content,
        timestamp=timestamp
    )
    await database.execute(query)

async def get_chat_history(conversation_id, limit=20):
    query = (
        chat_history.select()
        .where(chat_history.c.conversation_id == conversation_id)
        .order_by(chat_history.c.timestamp.desc())
        .limit(limit)
    )
    rows = await database.fetch_all(query)
    return list(reversed(rows))  # oldest first
