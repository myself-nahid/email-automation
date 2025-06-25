# --- START OF FILE chat_manager.py ---

from app.db import insert_chat_message, get_chat_history as db_get_chat_history

async def add_message(conversation_id, role, content):
    """Adds a message to the chat history. Timestamp is handled by the database."""
    await insert_chat_message(conversation_id, role, content)

async def get_chat_history(conversation_id, limit=20):
    """Retrieves the recent chat history for a conversation."""
    return await db_get_chat_history(conversation_id, limit)