# --- START OF FILE email_processor.py ---

import numpy as np
from app.openai_client import get_embedding
from datetime import datetime, timezone
from dateutil.parser import parse as parse_date
import logging

logger = logging.getLogger(__name__)

class ProcessEmailError(Exception):
    """Custom exception for email processing errors."""
    pass

def embedding_to_bytes(embedding_list: np.ndarray) -> bytes:
    if not isinstance(embedding_list, np.ndarray):
        embedding_list = np.array(embedding_list, dtype=np.float32)
    elif embedding_list.dtype != np.float32:
        embedding_list = embedding_list.astype(np.float32)
    return embedding_list.tobytes()

def bytes_to_embedding(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)

def extract_metadata(email_data: dict, account_id: str) -> dict:
    email_id = email_data.get('id', 'unknown_id')
    subject = email_data.get('subject', '')
    from_ = email_data.get('from') or email_data.get('from_', '')
    
    body_raw = email_data.get('body') or email_data.get('snippet', '')
    body = ' '.join(str(body_raw).split()) if body_raw else ''

    received_at_str = email_data.get('date') or email_data.get('received_at')
    is_unread_status = email_data.get('is_unread')
    received_at_dt = None

    if received_at_str:
        try:
            if isinstance(received_at_str, datetime):
                received_at_dt = received_at_str
            elif isinstance(received_at_str, (int, float)):
                 received_at_dt = datetime.fromtimestamp(received_at_str, tz=timezone.utc)
            elif isinstance(received_at_str, str):
                received_at_dt = parse_date(received_at_str)
            else:
                received_at_dt = datetime.now(timezone.utc)

            if received_at_dt.tzinfo is None:
                received_at_dt = received_at_dt.replace(tzinfo=timezone.utc)
            else:
                received_at_dt = received_at_dt.astimezone(timezone.utc)
        except Exception as e:
            logger.warning(f"Date parsing error for email {email_id}: {e}. Using current time.")
            received_at_dt = datetime.now(timezone.utc)
    else:
        received_at_dt = datetime.now(timezone.utc)
        
    category = "general"
    content_to_check = (str(subject) + " " + str(body)).lower()
    
    if any(kw in content_to_check for kw in ['meeting', 'interview', 'call', 'appointment', 'schedule', 'calendar', 'zoom', 'teams', 'webinar', 'event']):
        category = "meeting/event"
    elif any(kw in content_to_check for kw in ['urgent', 'asap', 'important', 'critical', 'action required', 'deadline']):
        category = "urgent"
    elif any(kw in content_to_check for kw in ['invoice', 'payment', 'receipt', 'order']):
        category = "financial"

    return {
        "email_id": email_id,
        "account_id": account_id,
        "subject": str(subject),
        "from_": str(from_),
        "body": str(body),
        "received_at": received_at_dt.isoformat(),
        "is_unread": is_unread_status,
        "category": category,
    }

async def process_email(email_data: dict, generate_embedding_only: bool = False) -> dict:
    email_id = email_data.get('id')
    account_id = email_data.get('account_id')

    if not email_id:
        raise ProcessEmailError("'id' is missing from email_data.")
    if not account_id and not generate_embedding_only:
        raise ProcessEmailError(f"'account_id' is missing for email {email_id}.")
    
    account_id = account_id or "temp_query_account"

    try:
        metadata = extract_metadata(email_data, account_id)
        
        if generate_embedding_only or not email_data.get('skip_embedding', False):
            text_for_embedding = f"Subject: {metadata.get('subject', '')}\nFrom: {metadata.get('from_', '')}\nBody: {metadata.get('body', '')[:2000]}"
            embedding_vector = await get_embedding(text_for_embedding)
            if embedding_vector is None:
                raise ProcessEmailError(f"Embedding generation failed for email {email_id}.")
            
            metadata['embedding'] = embedding_to_bytes(embedding_vector)
            metadata['full_text_for_embedding'] = text_for_embedding

        if generate_embedding_only:
            return {
                "email_id": metadata['email_id'],
                "embedding": metadata.get('embedding')
            }

        return metadata
        
    except Exception as e:
        logger.error(f"Error processing email {email_id}: {str(e)}", exc_info=True)
        if isinstance(e, ProcessEmailError):
            raise
        raise ProcessEmailError(f"Failed to process email {email_id}: {e}") from e