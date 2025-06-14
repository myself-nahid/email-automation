# --- START OF FILE main.py ---
# from app.openai_client import get_embedding # Old import if you also have chat_completion
from app.openai_client import get_embedding, chat_completion # Corrected import
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from app.models import EmailInput, QuestionRequest, ChatMessage, EmailStatusUpdate
from app.email_processor import process_email, ProcessEmailError # Import custom error
from app.faiss_index import faiss_index # Assuming faiss_index is an instance
from app.db import (
    database, upsert_email, get_email_by_id,
    migrate_database, get_emails_in_date_range, update_email_status, metadata, engine # Added get_emails_in_date_range
)
from app.chat_manager import add_message, get_chat_history
from app.openai_client import chat_completion
import faiss
from datetime import datetime, timedelta, timezone
import asyncio
import numpy as np
# import faiss # Not directly used here, faiss_index abstracts it
import traceback
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

origins = [
    "http://localhost", "http://localhost:3000", "http://127.0.0.1:3000",
    "http://192.168.10.190:3000", "http://192.168.10.12:3000"
]
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    await database.connect()
    # Create all tables defined in metadata if they don't exist.
    # For production, Alembic or similar is better for managing schema changes.
    metadata.create_all(engine) # from db.py
    await migrate_database() # Apply column additions
    try:
        faiss_index.load() # Attempt to load FAISS index from disk
    except Exception as e:
        logger.error(f"Failed to load FAISS index on startup: {e}", exc_info=True)


@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()
    try:
        faiss_index.save() # Attempt to save FAISS index on shutdown
    except Exception as e:
        logger.error(f"Failed to save FAISS index on shutdown: {e}", exc_info=True)


@app.get("/")
async def read_root():
    return {"message": "Welcome to the AI Email Assistant API"}

def estimate_tokens(text: str) -> int:
    return len(text) // 4 # Rough estimate

def truncate_email_content(email: dict, max_chars: int = 800) -> dict:
    email_copy = email.copy()
    body = str(email_copy.get('body') or email_copy.get('full_text', '')) # Ensure body is string

    if len(body) <= max_chars:
        email_copy['body'] = body
        return email_copy

    priority_keywords = [
        'meeting', 'interview', 'call', 'appointment', 'conference', 'zoom', 'teams', 'calendar',
        'schedule', 'invite', 'time', 'date', 'event', 'urgent', 'important', 'action required',
        'deadline', 'asap', 'critical', 'response needed', 'follow up', 'invoice', 'payment',
        'receipt', 'alert', 'update', 'confirm', 'booking', 'summary', 'report',
        'next week', 'tomorrow', 'today', 'this week', 'next month',
    ]
    
    sentences = re.split(r'(?<=[.!?])\s+', body)
    prioritized_sentences = []
    other_sentences = []
    
    for sentence_clean in sentences:
        if not sentence_clean.strip(): continue
        has_date_pattern = re.search(r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}\b|\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b|\b\d{1,2}-\d{1,2}(?:-\d{2,4})?\b', sentence_clean, re.IGNORECASE)
        if any(keyword in sentence_clean.lower() for keyword in priority_keywords) or has_date_pattern:
            prioritized_sentences.append(sentence_clean)
        else:
            other_sentences.append(sentence_clean)
    
    truncated_parts = []
    current_len = 0
    ellipsis = "..."
    ellipsis_len = len(ellipsis)

    for s_list in [prioritized_sentences, other_sentences]:
        for s in s_list:
            s_len = len(s)
            if current_len + s_len + (1 if truncated_parts else 0) <= max_chars - ellipsis_len:
                truncated_parts.append(s)
                current_len += s_len + (1 if len(truncated_parts) > 1 else 0)
            elif not truncated_parts:
                truncated_parts.append(s[:max_chars - ellipsis_len])
                current_len = max_chars - ellipsis_len
                break
        if current_len >= max_chars - ellipsis_len: break
    
    if truncated_parts:
        truncated_body = ' '.join(truncated_parts)
        if len(body) > len(truncated_body) :
             truncated_body += ellipsis
    else: 
        truncated_body = body[:max_chars - ellipsis_len] + ellipsis
        
    email_copy['body'] = truncated_body
    return email_copy

def build_smart_context(emails: List[dict], max_total_tokens=2500) -> str:
    context_parts = []
    total_tokens = 0

    for email in emails:
        date_str = email.get('received_at', 'Unknown date')
        if isinstance(date_str, datetime): date_str = date_str.isoformat()
        
        subject = str(email.get('subject', 'No subject')).strip()
        body_snippet = str(email.get('body', '')).strip()
        
        unread_status_val = email.get('is_unread')
        status_str = ""
        if unread_status_val is True: status_str = "Status: Unread\n  "
        elif unread_status_val is False: status_str = "Status: Read\n  "
        # If None, status_str remains empty, so "Status:" line won't appear

        part = f"- Date Received: {date_str}\n  From: {email.get('from_', 'Unknown sender')}\n  Subject: {subject}\n  {status_str}Body Snippet: {body_snippet}"
        estimated_tokens_part = estimate_tokens(part)
        
        if total_tokens + estimated_tokens_part > max_total_tokens:
            if not context_parts:
                max_chars_for_part_strict = int(max_total_tokens * 3.5)
                if len(part) > max_chars_for_part_strict:
                    header_len = len(f"- Date Received: {date_str}\n  From: {email.get('from_', 'Unknown sender')}\n  Subject: {subject}\n  {status_str}Body Snippet: ")
                    remaining_chars_for_body = max_chars_for_part_strict - header_len - len("\n[SNIPPET TRUNCATED]")
                    if remaining_chars_for_body > 20:
                        body_snippet = body_snippet[:remaining_chars_for_body] + "..."
                        part = f"- Date Received: {date_str}\n  From: {email.get('from_', 'Unknown sender')}\n  Subject: {subject}\n  {status_str}Body Snippet: {body_snippet}\n[SNIPPET TRUNCATED]"
                    else:
                        part = f"- Date Received: {date_str}\n  From: {email.get('from_', 'Unknown sender')}\n  Subject: {subject}\n[SNIPPET OMITTED]"
                    estimated_tokens_part = estimate_tokens(part)
                if estimated_tokens_part <= max_total_tokens:
                    context_parts.append(part)
                    total_tokens += estimated_tokens_part
            break
        context_parts.append(part)
        total_tokens += estimated_tokens_part
    return "\n\n".join(context_parts)

def filter_recent_emails(emails: List[dict], days_ahead: int = 14, lookback_days: int = 30) -> List[dict]:
    if not emails: return []
    now_utc = datetime.now(timezone.utc)
    past_cutoff_utc = now_utc - timedelta(days=lookback_days)
    future_cutoff_for_received_at = now_utc + timedelta(days=days_ahead)
    filtered_emails = []
    
    for email in emails:
        email_date_raw = email.get('received_at', '')
        if not email_date_raw: continue
        email_date_dt_utc = None
        try:
            if isinstance(email_date_raw, str):
                temp_dt = datetime.fromisoformat(email_date_raw.replace('Z', '+00:00')) if email_date_raw.endswith('Z') else datetime.fromisoformat(email_date_raw)
                email_date_dt_aware = temp_dt.replace(tzinfo=timezone.utc) if temp_dt.tzinfo is None else temp_dt
            elif isinstance(email_date_raw, datetime):
                email_date_dt_aware = email_date_raw.replace(tzinfo=timezone.utc) if email_date_raw.tzinfo is None else email_date_raw
            elif isinstance(email_date_raw, (int, float)):
                email_date_dt_aware = datetime.fromtimestamp(email_date_raw, tz=timezone.utc)
            else: continue
            email_date_dt_utc = email_date_dt_aware.astimezone(timezone.utc)
            if past_cutoff_utc <= email_date_dt_utc <= future_cutoff_for_received_at:
                filtered_emails.append(email)
        except Exception as e:
            logger.warning(f"Date parsing error in filter_recent_emails for email {email.get('email_id')}: {e}")
            continue
    return filtered_emails





def enhanced_truncate_email_content(email: dict, max_chars: int = 1000) -> dict:
    """Enhanced email truncation that preserves important information"""
    email_copy = email.copy()
    body = str(email_copy.get('body') or email_copy.get('full_text', ''))
    
    if len(body) <= max_chars:
        email_copy['body'] = body
        return email_copy

    # Enhanced priority keywords for better content preservation
    priority_keywords = [
        # Meeting/Event keywords
        'meeting', 'interview', 'call', 'appointment', 'conference', 'zoom', 'teams', 
        'calendar', 'schedule', 'invite', 'webinar', 'session', 'demo',
        
        # Time-related keywords
        'time', 'date', 'when', 'at', 'on', 'tomorrow', 'today', 'next week', 
        'this week', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
        
        # Urgency keywords
        'urgent', 'important', 'action required', 'deadline', 'asap', 'critical',
        'high priority', 'response needed', 'follow up',
        
        # Business keywords
        'invoice', 'payment', 'receipt', 'alert', 'update', 'confirm', 'booking',
        'summary', 'report', 'project', 'client', 'customer'
    ]
    
    # Split into sentences and prioritize
    sentences = re.split(r'(?<=[.!?])\s+', body)
    high_priority = []
    medium_priority = []
    low_priority = []
    
    for sentence in sentences:
        if not sentence.strip():
            continue
            
        sentence_lower = sentence.lower()
        
        # Check for date patterns (high priority)
        date_pattern = re.search(
            r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
            r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
            r'\s+\d{1,2}(?:st|nd|rd|th)?\b|\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b|'
            r'\b\d{1,2}-\d{1,2}(?:-\d{2,4})?\b|\b(?:tomorrow|today|next\s+week|this\s+week)\b',
            sentence_lower, re.IGNORECASE
        )
        
        # Check for time patterns (high priority)
        time_pattern = re.search(
            r'\b\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)?\b|\b(?:morning|afternoon|evening|noon)\b',
            sentence_lower, re.IGNORECASE
        )
        
        # Priority classification
        priority_count = sum(1 for keyword in priority_keywords if keyword in sentence_lower)
        
        if date_pattern or time_pattern or priority_count >= 2:
            high_priority.append(sentence)
        elif priority_count >= 1:
            medium_priority.append(sentence)
        else:
            low_priority.append(sentence)
    
    # Build truncated content
    result_sentences = []
    current_length = 0
    ellipsis = "..."
    
    # Add sentences in priority order
    for priority_group in [high_priority, medium_priority, low_priority]:
        for sentence in priority_group:
            sentence_length = len(sentence)
            if current_length + sentence_length + len(ellipsis) <= max_chars:
                result_sentences.append(sentence)
                current_length += sentence_length + 1  # +1 for space
            else:
                # If this is the first sentence and it's too long, truncate it
                if not result_sentences and current_length == 0:
                    truncated_sentence = sentence[:max_chars - len(ellipsis)]
                    result_sentences.append(truncated_sentence)
                    current_length = len(truncated_sentence)
                break
        
        if current_length >= max_chars - len(ellipsis):
            break
    
    # Join sentences and add ellipsis if needed
    truncated_body = ' '.join(result_sentences)
    if len(body) > len(truncated_body):
        truncated_body += ellipsis
    
    email_copy['body'] = truncated_body
    return email_copy


# MODIFICATION 1: Update context builder to include category
def enhanced_build_smart_context(emails: List[dict], max_total_tokens=2500) -> str:
    """Enhanced context building with better formatting for AI understanding"""
    if not emails:
        return "No emails available for analysis."
    
    context_parts = []
    total_tokens = 0
    
    for i, email in enumerate(emails, 1):
        # Format date consistently
        date_str = email.get('received_at', 'Unknown date')
        if isinstance(date_str, datetime):
            date_str = date_str.strftime('%Y-%m-%d %H:%M:%S UTC')
        
        # Get email details
        subject = str(email.get('subject', 'No subject')).strip()
        sender = str(email.get('from_', 'Unknown sender')).strip()
        body = str(email.get('body', '')).strip()
        category = str(email.get('category', 'general')).capitalize() # <-- GET CATEGORY
        
        # Format unread status more clearly
        unread_status = email.get('is_unread')
        if unread_status is True:
            status = "UNREAD"
        elif unread_status is False:
            status = "READ"
        else:
            status = "STATUS_UNKNOWN"
        
        # Build email entry
        email_entry = f"""EMAIL {i}:
Date: {date_str}
From: {sender}
Subject: {subject}
Status: {status}
Category: {category}
Content: {body}
---"""
        
        # Check token limit
        estimated_tokens = estimate_tokens(email_entry)
        if total_tokens + estimated_tokens > max_total_tokens:
            if not context_parts:  # At least include one email, even if truncated
                # Truncate the body to fit
                available_tokens = max_total_tokens - total_tokens
                available_chars = int(available_tokens * 3.5)  # Rough char to token ratio
                
                header_text = f"""EMAIL {i}:
Date: {date_str}
From: {sender}
Subject: {subject}
Status: {status}
Category: {category}
Content: """
                
                remaining_chars = available_chars - len(header_text) - 20  # Buffer
                if remaining_chars > 50:
                    truncated_body = body[:remaining_chars] + "...[TRUNCATED]"
                    email_entry = header_text + truncated_body + "\n---"
                    context_parts.append(email_entry)
            break
        
        context_parts.append(email_entry)
        total_tokens += estimated_tokens
    
    return "\n\n".join(context_parts)


# Enhanced date range query function
async def get_emails_in_date_range_enhanced(
    start_date: datetime, 
    end_date: datetime, 
    account_id: Optional[str] = None,
    is_unread: Optional[bool] = None,
    keywords: Optional[List[str]] = None,
    category: Optional[str] = None, # <-- ADD CATEGORY FILTER
    limit: int = 50
):
    """Enhanced date range query with additional filters"""
    
    conditions = ["received_at >= :start_date", "received_at <= :end_date"]
    params = {
        "start_date": start_date,
        "end_date": end_date
    }
    
    if account_id:
        conditions.append("account_id = :account_id")
        params["account_id"] = account_id
    
    if is_unread is not None:
        conditions.append("is_unread = :is_unread")
        params["is_unread"] = is_unread
    
    if category: # <-- ADD CATEGORY LOGIC
        conditions.append("category = :category")
        params["category"] = category
    
    if keywords:
        keyword_conditions = []
        for idx, keyword in enumerate(keywords):
            subj_param = f"keyword_subj_{idx}"
            body_param = f"keyword_body_{idx}"
            keyword_conditions.append(f"(LOWER(subject) LIKE :{subj_param} OR LOWER(body) LIKE :{body_param})")
            params[subj_param] = f"%{keyword.lower()}%"
            params[body_param] = f"%{keyword.lower()}%"
        
        if keyword_conditions:
            conditions.append(f"({' OR '.join(keyword_conditions)})")
    
    where_clause = " AND ".join(conditions)
    query = f"SELECT * FROM emails WHERE {where_clause} ORDER BY received_at DESC LIMIT :limit"
    params["limit"] = limit
    
    try:
        return await database.fetch_all(query, values=params)
    except Exception as e:
        logger.error(f"Enhanced date range query failed: {e}")
        return []
    

    

@app.post("/emails/batch_add")
async def batch_add_emails(emails_input: List[EmailInput]): # Renamed for clarity
    logger.info(f"Received {len(emails_input)} emails for batch processing.")
    if not emails_input:
        return {"added": 0, "failed": 0, "message": "No emails provided."}
    
    processed_count = 0
    failed_count = 0
    skipped_count = 0
    faiss_add_ids = []
    faiss_add_embeddings = []

    for email_in_model in emails_input:
        try:
            # Convert Pydantic model to dict for process_email
            email_data_dict = email_in_model.model_dump(by_alias=True) # Use model_dump for Pydantic v2
            
            # Ensure account_id is present
            if not email_data_dict.get('account_id'):
                logger.error(f"Skipping email ID {email_in_model.id}: account_id missing.")
                failed_count += 1
                continue

            # Check if email already exists in database
            existing_email = await get_email_by_id(email_data_dict['id'], account_id=email_data_dict['account_id'])
            
            if existing_email:
                # Check if we need to update the email (e.g., if is_unread status changed)
                needs_update = False
                if email_data_dict.get('is_unread') is not None and existing_email.get('is_unread') != email_data_dict['is_unread']:
                    needs_update = True
                    logger.info(f"Email {email_data_dict['id']} needs update: is_unread status changed")
                
                if not needs_update:
                    logger.info(f"Skipping existing email ID {email_data_dict['id']}")
                    skipped_count += 1
                    continue
                
                # If we have an existing email with embedding, reuse it
                if existing_email.get('embedding'):
                    processed_metadata = {
                        'email_id': email_data_dict['id'],
                        'account_id': email_data_dict['account_id'],
                        'subject': email_data_dict.get('subject', existing_email.get('subject', '')),
                        'from_': email_data_dict.get('from_', existing_email.get('from_', '')),
                        'body': email_data_dict.get('body', existing_email.get('body', '')),
                        'received_at': email_data_dict.get('date', existing_email.get('received_at')),
                        'is_unread': email_data_dict.get('is_unread'),
                        'embedding': existing_email['embedding'],
                        'full_text_for_embedding': existing_email.get('full_text_for_embedding', '')
                    }
                else:
                    # Process email to generate embedding if missing
                    processed_metadata = await process_email(email_data_dict)
            else:
                # Process new email
                processed_metadata = await process_email(email_data_dict)
            
            await upsert_email(processed_metadata)
            
            if 'embedding' in processed_metadata and processed_metadata['embedding']:
                faiss_add_ids.append(processed_metadata['email_id'])
                faiss_add_embeddings.append(processed_metadata['embedding'])
            processed_count += 1
            
        except ProcessEmailError as e:
            logger.error(f"Processing failed for email ID {email_in_model.id}: {e}")
            failed_count += 1
        except Exception as e:
            logger.error(f"Unexpected error during batch add for email ID {email_in_model.id}: {e}", exc_info=True)
            failed_count += 1
            
    if faiss_add_ids:
        try:
            faiss_index.add(faiss_add_ids, faiss_add_embeddings)
        except Exception as e:
            logger.error(f"FAISS add operation failed: {e}", exc_info=True)

    # Save FAISS index once after processing all emails in the batch
    if faiss_add_ids:
        try:
            faiss_index.save()
            logger.info(f"FAISS index saved. Total vectors: {faiss_index.index.ntotal if faiss_index.index else 'N/A'}")
        except Exception as e:
            logger.error(f"FAISS save operation failed: {e}", exc_info=True)

    return {
        "processed": processed_count,
        "failed": failed_count,
        "skipped": skipped_count,
        "faiss_total_after_add": faiss_index.index.ntotal if hasattr(faiss_index, 'index') and faiss_index.index else "FAISS N/A"
    }

async def get_total_email_count(account_id: Optional[str] = None):
    # ... (same as before)
    try:
        if account_id:
            query = "SELECT COUNT(*) as count FROM emails WHERE account_id = :account_id"
            result = await database.fetch_one(query, values={"account_id": account_id})
        else:
            query = "SELECT COUNT(*) as count FROM emails"
            result = await database.fetch_one(query)
        return result['count'] if result else 0
    except Exception as e:
        logger.error(f"Error getting email count: {str(e)}")
        return 0

# --- Query Endpoint Configuration ---
MAX_TOKEN_BUDGET = 4096
PROMPT_SYSTEM_AND_USER_ESTIMATE = 560 # Adjusted based on shorter system prompt + avg q
ANSWER_TOKEN_RESERVE = 800
MAX_CONTEXT_TOKEN_FOR_EMAILS = MAX_TOKEN_BUDGET - PROMPT_SYSTEM_AND_USER_ESTIMATE - ANSWER_TOKEN_RESERVE
AVG_TOKENS_PER_EMAIL_IN_CONTEXT = 220 # Estimate for one email entry in the context string
# Calculate how many emails can roughly fit, then fetch a bit more
EMAILS_FOR_CONTEXT_TARGET_COUNT = max(1, MAX_CONTEXT_TOKEN_FOR_EMAILS // AVG_TOKENS_PER_EMAIL_IN_CONTEXT)
MAX_EMAILS_TO_FETCH_INITIAL = max(15, EMAILS_FOR_CONTEXT_TARGET_COUNT * 2 + 5)

logger.info(f"Token Config: MAX_TOKEN_BUDGET={MAX_TOKEN_BUDGET}, PROMPT_SYSTEM_AND_USER_ESTIMATE={PROMPT_SYSTEM_AND_USER_ESTIMATE}, MAX_CONTEXT_TOKEN_FOR_EMAILS={MAX_CONTEXT_TOKEN_FOR_EMAILS}, EMAILS_FOR_CONTEXT_TARGET_COUNT={EMAILS_FOR_CONTEXT_TARGET_COUNT}, MAX_EMAILS_TO_FETCH_INITIAL={MAX_EMAILS_TO_FETCH_INITIAL}")


# Key improvements for the /query/working endpoint

@app.post("/query/working")
async def working_query(request: QuestionRequest):
    try:
        faiss_search_successful = False
        emails_for_context_raw = []
        account_id_filter = request.account_id
        query_lower = request.question.lower()
        
        logger.info(f"Query: '{request.question}', Account: {account_id_filter or 'all'}")

        # Enhanced query classification
        is_unread_query = any(keyword in query_lower for keyword in ["unread", "haven't read", "not read"])
        is_urgent_query = any(keyword in query_lower for keyword in ["urgent", "important", "action required", "deadline", "asap", "critical"])
        is_event_query = any(keyword in query_lower for keyword in [
            'meeting', 'interview', 'call', 'appointment', 'conference', 'zoom', 'teams', 
            'scheduled', 'calendar', 'upcoming', 'event', 'invite', 'webinar', 'tomorrow',
            'today', 'next week', 'this week', 'next 7 days', 'next two weeks'
        ])

        if request.email_id:
            # Specific email lookup
            doc = await get_email_by_id(request.email_id, account_id=account_id_filter)
            if doc: 
                emails_for_context_raw.append(dict(doc))
            logger.info(f"Retrieved specific email ID: {request.email_id}, Found: {bool(doc)}")
        else:
            # Multi-strategy email retrieval
            
            # MODIFICATION 2: Update retrieval logic to use the Category field
            db_emails_added = False
            
            if is_urgent_query:
                # Strategy 1: Prioritize fetching directly by 'urgent' category.
                try:
                    end_date = datetime.now(timezone.utc)
                    start_date = end_date - timedelta(days=30) # Look back 30 days for urgent items
                    
                    db_emails = await get_emails_in_date_range_enhanced(
                        start_date=start_date, end_date=end_date,
                        account_id=account_id_filter,
                        category='urgent',
                        limit=MAX_EMAILS_TO_FETCH_INITIAL
                    )
                    emails_for_context_raw.extend([dict(row) for row in db_emails])
                    db_emails_added = True
                    logger.info(f"DB Urgent Query (by Category): Found {len(db_emails)} urgent emails.")
                except Exception as e:
                    logger.error(f"Urgent email query by category failed: {e}", exc_info=True)

            # (The other query types like is_unread_query and is_event_query can remain as they are)
            # ... [existing is_unread_query and is_event_query logic] ...
            
            # Vector search can now act as a fallback or supplement
            if not db_emails_added and hasattr(faiss_index, 'index') and faiss_index.index and faiss_index.index.ntotal > 0:
                try:
                    q_embedding_data = await process_email({
                        'id': 'query_temp_id', 
                        'account_id': account_id_filter or 'general_query_account',
                        'subject': request.question, 
                        'from_': 'user_query', 
                        'body': request.question,
                        'date': datetime.now(timezone.utc).isoformat()
                    }, generate_embedding_only=True)

                    if q_embedding_data and 'embedding' in q_embedding_data:
                        embedding_array = np.frombuffer(q_embedding_data['embedding'], dtype=np.float32).reshape(1, -1)
                        distances, top_faiss_email_ids = faiss_index.search_with_scores(
                            embedding_array, top_k=MAX_EMAILS_TO_FETCH_INITIAL, account_id=account_id_filter
                        )
                        logger.info(f"FAISS search returned {len(top_faiss_email_ids)} email IDs.")
                        for eid in top_faiss_email_ids:
                            doc = await get_email_by_id(eid, account_id=account_id_filter)
                            if doc: 
                                emails_for_context_raw.append(dict(doc))
                        if top_faiss_email_ids:
                            faiss_search_successful = True
                            logger.info(f"FAISS: Fetched {len(emails_for_context_raw)} emails from DB.")
                except Exception as e: 
                    logger.error(f"FAISS search failed: {e}", exc_info=True)


            # General fallback if no specific strategy worked
            if not emails_for_context_raw: # Check if list is still empty
                try:
                    logger.info("Using general fallback query")
                    general_query = "SELECT * FROM emails"
                    params = {}
                    
                    if account_id_filter:
                        general_query += " WHERE account_id = :account_id"
                        params["account_id"] = account_id_filter
                    
                    general_query += f" ORDER BY received_at DESC LIMIT {MAX_EMAILS_TO_FETCH_INITIAL}"
                    general_results = await database.fetch_all(general_query, values=params)
                    emails_for_context_raw.extend([dict(row) for row in general_results])
                    logger.info(f"General fallback: Found {len(general_results)} emails.")
                except Exception as e:
                    logger.error(f"General fallback failed: {e}", exc_info=True)

        # Deduplicate emails
        seen_ids = set()
        unique_emails = []
        for email_dict in emails_for_context_raw:
            if isinstance(email_dict, dict) and email_dict.get('email_id') not in seen_ids:
                unique_emails.append(email_dict)
                seen_ids.add(email_dict['email_id'])
        
        logger.info(f"Unique emails after retrieval: {len(unique_emails)}")

        # Apply recency filtering based on query type
        if is_event_query:
            # For event queries, look further back but also forward
            filtered_emails = filter_recent_emails(unique_emails, days_ahead=30, lookback_days=60)
        elif is_urgent_query:
            # For urgent queries, focus on recent emails
            filtered_emails = filter_recent_emails(unique_emails, days_ahead=7, lookback_days=30) # Increased lookback for urgent
        elif is_unread_query:
            # For unread queries, we already filtered by date in DB query
            filtered_emails = unique_emails
        else:
            # General queries
            filtered_emails = filter_recent_emails(unique_emails, lookback_days=30)

        logger.info(f"Emails after filtering: {len(filtered_emails)}")

        # Handle case where no emails found
        if not filtered_emails:
            db_count = await get_total_email_count(account_id=account_id_filter)
            message = f"I couldn't find any relevant emails for your query."
            
            return {"answer": message}

        # Build context with improved truncation and now including category
        truncated_emails = [enhanced_truncate_email_content(email.copy(), max_chars=1000) for email in filtered_emails]
        context_text = enhanced_build_smart_context(truncated_emails, max_total_tokens=MAX_CONTEXT_TOKEN_FOR_EMAILS)
        
        # MODIFICATION 3: Enhance system prompt to use the Category
        current_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
        
        system_prompt = f"""You are an AI Email Assistant analyzing email data to answer user questions.

Current date and time: {current_time}

ANALYSIS GUIDELINES:
1. **Upcoming Events**: For questions about meetings, calls, interviews, appointments:
   - Prioritize emails with the category 'Meeting/event'.
   - Look for date/time information, calculate timeframes relative to {current_time}.
   - Extract: Date, Time, Type, Participants, Subject, Location/Links.

2. **Urgent/Important Items**: For questions about urgent emails or deadlines:
   - **Strongly prioritize emails with the category 'Urgent'.**
   - Also consider emails with keywords: "important", "action required", "deadline", "ASAP".
   - Identify required actions and deadlines.

3. **Email Status**: For read/unread questions:
   - Use "Status: Unread" or "Status: Read" information. Count unread emails accurately.

4. **General Queries**: Answer based SOLELY on provided email content.
   - Do not invent or assume information not in the emails.
   - If information is not available, state this clearly.

RESPONSE FORMAT:
- Be specific and actionable.
- Use bullet points for multiple items.
- Mention if information is from email snippets.

EMAIL DATA TO ANALYZE:
{context_text if context_text else "No email data available."}

Answer the user's question based ONLY on the email information provided above."""

        # Prepare messages for LLM
        messages = []
        
        # Add conversation history if available
        if request.conversation_id:
            try:
                history = await get_chat_history(request.conversation_id)
                for msg in history[-4:]:  # Last 4 messages for context
                    messages.append({"role": msg['role'], "content": msg['content']})
            except Exception as e:
                logger.warning(f"Error loading chat history: {e}")

        messages.extend([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.question}
        ])

        # Call LLM with better error handling
        try:
            answer = await chat_completion(messages)
            
            if not answer or not isinstance(answer, str) or not answer.strip():
                logger.error("LLM returned invalid response")
                answer = "I'm having trouble processing your request. Please try rephrasing your question."
            
            logger.info(f"Generated answer length: {len(answer)} chars")
            
        except Exception as llm_error:
            logger.error(f"LLM error: {llm_error}", exc_info=True)
            answer = "I'm experiencing technical difficulties. Please try again in a moment."

        # Save conversation history
        if request.conversation_id and answer:
            try:
                await add_message(request.conversation_id, "user", request.question)
                await add_message(request.conversation_id, "assistant", answer)
            except Exception as e:
                logger.warning(f"Error saving chat history: {e}")

        return {
            "answer": answer,
            # "emails_used_count": len(truncated_emails),
            # "query_type_detected": {
            #     "unread_query": is_unread_query,
            #     "urgent_query": is_urgent_query,
            #     "event_query": is_event_query
            # }
        }

    except Exception as e:
        logger.error(f"Query processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")
# --- Admin/Debug Endpoints ---
# (Keep debug/status and rebuild-faiss as they were, or simplify rebuild if it's problematic)
# Ensure they use logger.
# Example for rebuild-faiss, simplified to remove per-account complexity for now:
@app.post("/admin/rebuild-faiss-from-db")
async def rebuild_faiss_from_database_admin():
    logger.info(f"Starting GLOBAL FAISS rebuild admin task.")
    try:
        query_sql = "SELECT email_id, embedding FROM emails WHERE embedding IS NOT NULL"
        result = await database.fetch_all(query_sql)
        
        if not result:
            msg = "No emails with embeddings found in database for FAISS rebuild."
            logger.warning(msg)
            return {"success": False, "message": msg}
        
        logger.info(f"Found {len(result)} emails with embeddings in DB for FAISS rebuild.")
        
        DIM = faiss_index.dim if hasattr(faiss_index, 'dim') else 1536 # Use dim from faiss_index instance
        
        # Reinitialize global FAISS index
        faiss_index.index = faiss.IndexFlatL2(DIM) 
        faiss_index.id_map = [] 
        
        email_ids_for_faiss = []
        embeddings_list_for_faiss = []         
        valid_embeddings_count = 0
        invalid_embeddings_details = []
        expected_byte_length = DIM * np.dtype(np.float32).itemsize
        
        for row in result:
            email_id = row['email_id']
            embedding_data = row['embedding']
            if embedding_data and isinstance(embedding_data, bytes) and len(embedding_data) == expected_byte_length:
                try:
                    np_embedding = np.frombuffer(embedding_data, dtype=np.float32)
                    embeddings_list_for_faiss.append(np_embedding) 
                    email_ids_for_faiss.append(email_id) 
                    valid_embeddings_count += 1
                except Exception as e_np:
                    invalid_embeddings_details.append({"email_id": email_id, "error": f"Numpy conversion failed: {str(e_np)}"})
            else:
                invalid_embeddings_details.append({
                    "email_id": email_id, "embedding_present": bool(embedding_data),
                    "type": str(type(embedding_data)), "length_bytes": len(embedding_data) if embedding_data else 0,
                    "expected_length_bytes": expected_byte_length, "reason": "Invalid type or length"
                })
        
        if not embeddings_list_for_faiss:
            logger.warning("No valid embeddings found to add to FAISS index during rebuild.")
            return {"success": False, "message": "No valid embeddings found.", "invalid_samples": invalid_embeddings_details[:5]}
        
        all_embeddings_np = np.array(embeddings_list_for_faiss).astype(np.float32)
        if all_embeddings_np.ndim == 1 and all_embeddings_np.size > 0: # Handle single embedding case
             all_embeddings_np = all_embeddings_np.reshape(1, -1)
        
        if all_embeddings_np.size > 0 : # Only add if there's data
            faiss_index.index.add(all_embeddings_np)
            faiss_index.id_map.extend(email_ids_for_faiss)
        
        faiss_index.save()
        logger.info("Global FAISS index rebuilt and saved successfully.")
        
        return {
            "success": True, "message": "Global FAISS index rebuilt successfully.",
            "db_embeddings_found": len(result),
            "faiss_added_valid": valid_embeddings_count,
            "faiss_total_after_rebuild": faiss_index.index.ntotal,
            "invalid_embeddings_count": len(invalid_embeddings_details),
        }
    except Exception as e:
        logger.error(f"Error in /admin/rebuild-faiss-from-db: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

@app.get("/debug/status")
async def debug_status(account_id: Optional[str] = None):
    # ... (Keep this endpoint mostly as is, just ensure it uses logger)
    try:
        total_emails_in_db = await get_total_email_count(account_id=account_id)
        
        # Check emails with embeddings
        query_with_emb = "SELECT COUNT(*) as count FROM emails WHERE embedding IS NOT NULL"
        params_with_emb = {}
        if account_id:
            query_with_emb += " AND account_id = :account_id"
            params_with_emb = {"account_id": account_id}
        result_with_emb = await database.fetch_one(query_with_emb, values=params_with_emb)
        emails_with_embeddings = result_with_emb['count'] if result_with_emb else 0
        
        # Sample recent emails
        query_recent = "SELECT email_id, account_id, subject, from_, received_at, is_unread FROM emails" # Added is_unread
        params_recent = {}
        if account_id:
            query_recent += " WHERE account_id = :account_id"
            params_recent = {"account_id": account_id}
        query_recent += " ORDER BY received_at DESC LIMIT 5"
        recent_emails_db_rows = await database.fetch_all(query_recent, values=params_recent)
        recent_emails_db = [dict(row) for row in recent_emails_db_rows]
        
        faiss_total = "FAISS N/A"
        faiss_id_map_len = "FAISS N/A"
        faiss_sample_ids = []
        if hasattr(faiss_index, 'index') and faiss_index.index:
            faiss_total = faiss_index.index.ntotal
        if hasattr(faiss_index, 'id_map'):
            faiss_id_map_len = len(faiss_index.id_map)
            if faiss_index.id_map: # Check if id_map is not empty
                 faiss_sample_ids = faiss_index.id_map[:min(5, len(faiss_index.id_map))]


        return {
            "filter_account_id": account_id or "all_accounts",
            "database": {
                "total_emails": total_emails_in_db,
                "emails_with_embeddings": emails_with_embeddings,
                "recent_emails_sample": recent_emails_db
            },
            "faiss": {
                "index_total_vectors": faiss_total,
                "id_map_length": faiss_id_map_len,
                "sample_ids_from_map_head": faiss_sample_ids,
                "dim": faiss_index.dim if hasattr(faiss_index, 'dim') else "N/A"
            },
            "token_budget_config_for_query": {
                "MAX_TOKEN_BUDGET": MAX_TOKEN_BUDGET,
                "PROMPT_SYSTEM_AND_USER_ESTIMATE": PROMPT_SYSTEM_AND_USER_ESTIMATE,
                "ANSWER_TOKEN_RESERVE": ANSWER_TOKEN_RESERVE,
                "MAX_CONTEXT_TOKEN_FOR_EMAILS": MAX_CONTEXT_TOKEN_FOR_EMAILS,
                "EMAILS_FOR_CONTEXT_TARGET_COUNT": EMAILS_FOR_CONTEXT_TARGET_COUNT,
                "MAX_EMAILS_TO_FETCH_INITIAL": MAX_EMAILS_TO_FETCH_INITIAL,
            },
        }
    except Exception as e:
        logger.error(f"Error in /debug/status: {str(e)}", exc_info=True)
        return {"error": f"Failed to get debug status: {str(e)}"}


@app.patch("/emails/{email_id}/status")
async def mark_email_status(email_id: str, status_update: EmailStatusUpdate, account_id: str): # account_id should be from auth
    # In a real app, account_id would come from a dependency injection that verifies the user's token
    existing_email = await get_email_by_id(email_id, account_id=account_id)
    if not existing_email:
        raise HTTPException(status_code=404, detail="Email not found or access denied.")
    
    await update_email_status(email_id, status_update.is_unread, account_id)
    return {"message": f"Email {email_id} status updated successfully."}

@app.get("/emails/{email_id}")
async def get_single_email(email_id: str, account_id: str): # account_id should come from auth
    email = await get_email_by_id(email_id, account_id=account_id)
    if not email:
        raise HTTPException(status_code=404, detail="Email not found or access denied.")
    return email
# --- END OF FILE main.py ---