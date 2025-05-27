from app.db import insert_chat_message, get_chat_history as db_get_chat_history
import time

async def add_message(conversation_id, role, content):
    timestamp = time.time()
    await insert_chat_message(conversation_id, role, content, timestamp)

async def get_chat_history(conversation_id, limit=20):
    return await db_get_chat_history(conversation_id, limit)