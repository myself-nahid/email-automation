from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from app.models import EmailInput, QuestionRequest, ChatMessage
from app.email_processor import process_email
from app.faiss_index import faiss_index
from app.db import (
    database, upsert_email, get_email_by_id, 
    get_emails_in_date_range, 
    migrate_database
)
# Use the real functions from your other files
from app.chat_manager import add_message, get_chat_history
from app.openai_client import chat_completion

from datetime import datetime, timedelta
import asyncio
import numpy as np
import faiss 
import traceback
import re

app = FastAPI()

# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://192.168.10.190:3000",
    "http://192.168.10.12:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    await database.connect()
    await migrate_database()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

@app.get("/")
async def read_root():
    return {"message": "Welcome to the AI Email Assistant API"}

# Token estimation function
def estimate_tokens(text: str) -> int:
    """Rough estimation: ~4 characters per token for English text"""
    return len(text) // 4

def truncate_email_content(email: dict, max_chars: int = 800) -> dict:
    """Truncate email content to fit within token limits while preserving key info"""
    email_copy = email.copy()
    
    # Always preserve these fields in full
    preserve_fields = ['email_id', 'account_id', 'subject', 'from_', 'received_at']
    
    # Get body content
    body = email_copy.get('body') or email_copy.get('full_text', '')
    
    if len(body) > max_chars:
        # Try to find the most important part of the email
        # Look for scheduling/meeting keywords first
        meeting_keywords = [
            'meeting', 'interview', 'call', 'appointment', 'conference', 
            'zoom', 'teams', 'calendar', 'schedule', 'invite', 'time', 'date'
        ]
        
        # Split into sentences and prioritize those with meeting keywords
        sentences = re.split(r'[.!?]+', body)
        important_sentences = []
        other_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if any(keyword in sentence.lower() for keyword in meeting_keywords):
                important_sentences.append(sentence)
            else:
                other_sentences.append(sentence)
        
        # Start with important sentences
        truncated = '. '.join(important_sentences)
        
        # Add other sentences if we have space
        remaining_chars = max_chars - len(truncated)
        if remaining_chars > 100:  # Only if we have decent space left
            additional_text = '. '.join(other_sentences)
            if len(additional_text) <= remaining_chars:
                truncated += '. ' + additional_text
            else:
                truncated += '. ' + additional_text[:remaining_chars-20] + '...'
        
        # If truncated is still empty or too short, just take the beginning
        if len(truncated) < 50:
            truncated = body[:max_chars] + '...' if len(body) > max_chars else body
            
        email_copy['body'] = truncated
        if 'full_text' in email_copy:
            email_copy['full_text'] = truncated
    
    return email_copy

def build_smart_context(emails: List[dict], max_total_tokens=3000) -> str:
    """
    Build a concise, structured context string from emails focusing on
    date, subject, and a short snippet from the body to keep token count manageable.
    """
    context_parts = []
    total_tokens = 0

    for email in emails:
        date = email.get('received_at', 'Unknown date')
        subject = email.get('subject', 'No subject').strip()
        body = email.get('body', '').strip().replace('\n', ' ').replace('\r', ' ')

        # Take first 200 chars for snippet to limit length
        snippet = (body[:200] + '...') if len(body) > 200 else body

        part = f"- Date: {date}\n  Subject: {subject}\n  Snippet: {snippet}"
        
        # Rough token estimate: 1 token ~ 4 chars (safe overestimate)
        estimated_tokens = len(part) // 4
        
        if total_tokens + estimated_tokens > max_total_tokens:
            break

        context_parts.append(part)
        total_tokens += estimated_tokens

    return "\n\n".join(context_parts)


from datetime import datetime, timedelta
from typing import List

def filter_recent_emails(emails: List[dict], days_ahead: int = 7) -> List[dict]:
    """Filter emails to focus on those likely to contain upcoming events"""
    if not emails:
        return emails
    
    now = datetime.utcnow()
    cutoff_date = now - timedelta(days=30)  # Look at emails from last 30 days
    future_cutoff = now + timedelta(days=days_ahead)
    
    filtered_emails = []
    
    for email in emails:
        email_date_raw = email.get('received_at', '')
        if not email_date_raw:
            continue
        
        email_date = None
        try:
            if isinstance(email_date_raw, str):
                # Handle 'Z' timezone designator in ISO format
                if 'Z' in email_date_raw:
                    email_date = datetime.fromisoformat(email_date_raw.replace('Z', '+00:00'))
                else:
                    email_date = datetime.fromisoformat(email_date_raw)
            elif isinstance(email_date_raw, (int, float)):
                # Assume Unix timestamp in seconds
                email_date = datetime.utcfromtimestamp(email_date_raw)
            else:
                # Unknown type, skip
                continue
            
            # Make naive datetime for comparison (remove tzinfo)
            if email_date.tzinfo is not None:
                email_date = email_date.replace(tzinfo=None)
            
            # Filter emails recent enough or possibly with upcoming events
            if cutoff_date <= email_date <= future_cutoff:
                filtered_emails.append(email)
        
        except Exception as e:
            # On parsing error, log and skip or include email
            print(f"Error parsing email date '{email_date_raw}': {e}")
            # Optionally include to be safe:
            # filtered_emails.append(email)
            continue
    
    return filtered_emails


# (The /emails/batch_add endpoint remains the same as previously provided)
@app.post("/emails/batch_add")
async def batch_add_emails(emails: List[EmailInput]):
    print(f"Received {len(emails)} emails for batch processing.")
    if not emails:
        return {"added": 0, "failed": 0, "message": "No emails provided."}
    
    email_ids_for_faiss = []
    embeddings_bytes_for_faiss = []
    successful_emails_info = []
    failed_emails_info = []
    
    for email_input in emails:
        try:
            email_data_dict = email_input.dict() 
            processed_email_metadata = await process_email(email_data_dict)
            await upsert_email(processed_email_metadata)
            
            email_ids_for_faiss.append(processed_email_metadata['email_id'])
            embeddings_bytes_for_faiss.append(processed_email_metadata['embedding'])
            successful_emails_info.append({
                "email_id": email_input.id, 
                "account_id": email_input.account_id
            })
            
        except Exception as e:
            failed_emails_info.append({
                "email_id": email_input.id, 
                "account_id": email_input.account_id,
                "error": str(e)
            })
            continue 

    if email_ids_for_faiss and embeddings_bytes_for_faiss:
        try:
            faiss_index.add(email_ids_for_faiss, embeddings_bytes_for_faiss)
        except Exception as e:
            return {
                "added_to_db": len(successful_emails_info),
                "failed_processing": len(failed_emails_info),
                "successful_emails": successful_emails_info,
                "failed_emails": failed_emails_info,
                "faiss_error": str(e),
                "message": "Emails processed and saved to DB, but FAISS indexing failed."
            }

    return {
        "added_to_db": len(successful_emails_info),
        "failed_processing": len(failed_emails_info),
        "successful_emails": successful_emails_info,
        "failed_emails": failed_emails_info,
        "faiss_total_after_add": faiss_index.index.ntotal if hasattr(faiss_index, 'index') else "FAISS not initialized"
    }


async def get_total_email_count(account_id: Optional[str] = None):
    try:
        if account_id:
            query = "SELECT COUNT(*) as count FROM emails WHERE account_id = :account_id"
            result = await database.fetch_one(query, values={"account_id": account_id})
        else:
            query = "SELECT COUNT(*) as count FROM emails"
            result = await database.fetch_one(query)
        return result['count'] if result else 0
    except Exception as e:
        print(f"Error getting email count: {str(e)}")
        return 0


MAX_TOKEN_BUDGET = 4000
PROMPT_TOKEN_ESTIMATE = 500
ANSWER_TOKEN_RESERVE = 1000
TOKENS_PER_EMAIL_ESTIMATE = 100

@app.post("/query/working")
async def working_query(request: QuestionRequest):
    try:
        faiss_search_successful = False
        emails_for_context = []
        account_id_filter = request.account_id
        
        # print(f"Query received: '{request.question}' for account: {account_id_filter or 'all'}")

        # Calculate max emails to include based on token budget
        max_emails_to_use = (MAX_TOKEN_BUDGET - PROMPT_TOKEN_ESTIMATE - ANSWER_TOKEN_RESERVE) // TOKENS_PER_EMAIL_ESTIMATE
        max_emails_to_use = max(10, max_emails_to_use)  # at least 10 emails

        if request.email_id:
            doc = await get_email_by_id(request.email_id, account_id=account_id_filter)
            if doc:
                emails_for_context.append(dict(doc))
        else:
            # Try FAISS vector search first
            if hasattr(faiss_index, 'index') and faiss_index.index.ntotal > 0:
                try:
                    q_embedding_data = await process_email({
                        'id': 'query_temp_id',
                        'account_id': account_id_filter or 'general_query_account',
                        'subject': '', 'from_': '', 'body': request.question,
                        'date': datetime.utcnow().isoformat()
                    })
                    top_k = 100  # Retrieve more, but we will limit later
                    top_ids = faiss_index.search(np.frombuffer(q_embedding_data['embedding'], dtype=np.float32), top_k=top_k)

                    print(f"FAISS search returned {len(top_ids)} email IDs")

                    for eid in top_ids[:max_emails_to_use]:
                        doc = await get_email_by_id(eid, account_id=account_id_filter)
                        if doc:
                            emails_for_context.append(dict(doc))

                    if emails_for_context:
                        faiss_search_successful = True
                        print(f"FAISS search successful: found {len(emails_for_context)} emails")
                    else:
                        print("FAISS returned IDs but no matching emails found in DB")

                except Exception as e:
                    print(f"FAISS search failed: {str(e)}")

            # If FAISS search failed or no emails found, fallback to DB query
            if not faiss_search_successful:
                # print("Using enhanced database fallback for query.")
                try:
                    query_lower = request.question.lower()

                    meeting_keywords = ['meeting', 'interview', 'call', 'appointment', 'conference', 'zoom', 'teams', 'scheduled', 'calendar', 'upcoming']
                    is_meeting_query = any(keyword in query_lower for keyword in meeting_keywords)

                    conditions = []
                    params = {}

                    if account_id_filter:
                        conditions.append("account_id = :account_id")
                        params["account_id"] = account_id_filter

                    if is_meeting_query:
                        meeting_condition = """(
                            LOWER(subject) LIKE :meeting1 OR LOWER(body) LIKE :meeting2 OR
                            LOWER(subject) LIKE :interview1 OR LOWER(body) LIKE :interview2 OR
                            LOWER(subject) LIKE :call1 OR LOWER(body) LIKE :call2 OR
                            LOWER(subject) LIKE :appointment1 OR LOWER(body) LIKE :appointment2 OR
                            LOWER(subject) LIKE :conference1 OR LOWER(body) LIKE :conference2 OR
                            LOWER(subject) LIKE :zoom1 OR LOWER(body) LIKE :zoom2 OR
                            LOWER(subject) LIKE :teams1 OR LOWER(body) LIKE :teams2 OR
                            LOWER(subject) LIKE :calendar1 OR LOWER(body) LIKE :calendar2 OR
                            LOWER(subject) LIKE :schedule1 OR LOWER(body) LIKE :schedule2 OR
                            LOWER(subject) LIKE :invite1 OR LOWER(body) LIKE :invite2
                        )"""
                        conditions.append(meeting_condition)
                        params.update({
                            'meeting1': '%meeting%', 'meeting2': '%meeting%',
                            'interview1': '%interview%', 'interview2': '%interview%',
                            'call1': '%call%', 'call2': '%call%',
                            'appointment1': '%appointment%', 'appointment2': '%appointment%',
                            'conference1': '%conference%', 'conference2': '%conference%',
                            'zoom1': '%zoom%', 'zoom2': '%zoom%',
                            'teams1': '%teams%', 'teams2': '%teams%',
                            'calendar1': '%calendar%', 'calendar2': '%calendar%',
                            'schedule1': '%schedule%', 'schedule2': '%schedule%',
                            'invite1': '%invite%', 'invite2': '%invite%'
                        })
                        limit = max_emails_to_use * 2  # fetch more to have buffer
                        order_by = "received_at DESC"
                    else:
                        conditions.append("received_at >= datetime('now', '-14 days')")
                        limit = max_emails_to_use * 2
                        order_by = "received_at DESC"

                    where_clause = " AND ".join(conditions) if conditions else "1=1"

                    fallback_query_sql = f"""
                        SELECT * FROM emails
                        WHERE {where_clause}
                        ORDER BY {order_by}
                        LIMIT {limit}
                    """

                    # print(f"Database fallback query: {fallback_query_sql}")
                    # print(f"Query parameters: {params}")

                    fallback_results = await database.fetch_all(fallback_query_sql, values=params)

                    # Optional: sort or filter fallback_results if needed here

                    # Limit to max_emails_to_use
                    emails_for_context.extend([dict(row) for row in fallback_results[:max_emails_to_use]])

                    # print(f"Database fallback found {len(emails_for_context)} emails")

                except Exception as e:
                    print(f"Database fallback failed: {str(e)}")
                    traceback.print_exc()

        # Remove duplicates
        seen_ids = set()
        unique_emails = []
        for email in emails_for_context:
            if email['email_id'] not in seen_ids:
                unique_emails.append(email)
                seen_ids.add(email['email_id'])
        emails_for_context = unique_emails

        # print(f"Unique emails before filtering: {len(emails_for_context)}")

        # Filter emails for recency and relevance
        emails_for_context = filter_recent_emails(emails_for_context)

        # print(f"Final email context count after filtering: {len(emails_for_context)}")

        if not emails_for_context:
            db_email_count = await get_total_email_count(account_id=account_id_filter)

            try:
                broad_query = "SELECT * FROM emails"
                broad_params = {}
                if account_id_filter:
                    broad_query += " WHERE account_id = :account_id"
                    broad_params["account_id"] = account_id_filter
                broad_query += " ORDER BY received_at DESC LIMIT 10"

                broad_results = await database.fetch_all(broad_query, values=broad_params)
                emails_for_context = [dict(row) for row in broad_results]

                if emails_for_context:
                    print(f"Broad search found {len(emails_for_context)} emails for general context")

            except Exception as e:
                print(f"Broad search failed: {str(e)}")

        if not emails_for_context:
            return {
                "answer": f"I couldn't find any emails to analyze. Database shows {db_email_count} total emails for account '{account_id_filter or 'all'}'. This might indicate an issue with email processing or database connectivity.",
                "emails": []
            }

        context_text = build_smart_context(emails_for_context, max_total_tokens=3000)

        current_date = datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M')

        system_prompt = f"""
You are an AI email assistant analyzing emails for upcoming meetings and events. You can also give some basic questions like Hello, How are you etc.

Current date and time: {current_time}

TASK:
- Find upcoming meetings, interviews, or calls within the next 7 days by analyzing ONLY the emails provided below.
- Find upcoming meetings, interviews, or calls within the next 14 days by analyzing ONLY the emails provided below.
- Find upcoming meetings, interviews, or calls within the next 21 days by analyzing ONLY the emails provided below.
- Find upcoming meetings, interviews, or calls within the next 28 days by analyzing ONLY the emails provided below.
- Find upcoming meetings, interviews, or calls within the next 15 days by analyzing ONLY the emails provided below.
- Find upcoming meetings, interviews, or calls within the next 30 days by analyzing ONLY the emails provided below.
- Find upcoming meetings, interviews, or calls within the next 3 days by analyzing ONLY the emails provided below.

IMPORTANT:
- Only use the provided emails for your answer.
- Do NOT guess or assume info not present in these emails.
- Provide clear event details including: Date, Time, Type (meeting/interview/call), Participants, Subject, Location or Link.
- Use ISO 8601 format for dates and times if possible.
- Use bullet points if multiple events.
- If no upcoming meetings or calls are found, respond: "After analyzing your emails, I found no upcoming meetings, interviews, or calls scheduled for the next 7 days."

EMAILS TO ANALYZE:
{context_text}
"""

        total_estimated_tokens = estimate_tokens(system_prompt) + estimate_tokens(request.question)
        # print(f"Estimated tokens for request: {total_estimated_tokens}")

        messages_for_ai = []
        if request.conversation_id:
            try:
                history = await get_chat_history(request.conversation_id)
                recent_history = history[-4:] if len(history) > 4 else history
                for m in recent_history:
                    messages_for_ai.append({"role": m['role'], "content": m['content']})
            except Exception as e:
                print(f"Error loading chat history: {str(e)}")

        messages_for_ai.append({"role": "system", "content": system_prompt})
        messages_for_ai.append({"role": "user", "content": request.question})

        total_message_tokens = sum(estimate_tokens(msg["content"]) for msg in messages_for_ai)
        # print(f"Final estimated tokens: {total_message_tokens}")

        answer = await chat_completion(messages_for_ai)

        if request.conversation_id:
            try:
                await add_message(request.conversation_id, "user", request.question)
                await add_message(request.conversation_id, "assistant", answer)
            except Exception as e:
                print(f"Error saving chat history: {str(e)}")

        return {
            "answer": answer,
            # "debug_info": {
            #     "account_queried": account_id_filter or "all",
            #     "emails_found": len(emails_for_context),
            #     "estimated_tokens": total_message_tokens,
            #     "faiss_total": faiss_index.index.ntotal if hasattr(faiss_index, 'index') else "N/A",
            #     "search_method": "faiss" if faiss_search_successful else "database_fallback"
            # }
        }

    except Exception as e:
        print(f"Query error in /query/working: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# --- Admin/Debug Endpoints ---

@app.get("/debug/status")
async def debug_status(account_id: Optional[str] = None):
    """Comprehensive debug status, optionally filtered by account_id."""
    try:
        total_emails_in_db = await get_total_email_count(account_id=account_id)
        
        # Check emails with embeddings
        if account_id:
            query_with_emb = "SELECT COUNT(*) as count FROM emails WHERE embedding IS NOT NULL AND account_id = :account_id"
            params_with_emb = {"account_id": account_id}
        else:
            query_with_emb = "SELECT COUNT(*) as count FROM emails WHERE embedding IS NOT NULL"
            params_with_emb = {}
        result_with_emb = await database.fetch_one(query_with_emb, values=params_with_emb)
        emails_with_embeddings = result_with_emb['count'] if result_with_emb else 0
        
        # Sample recent emails
        if account_id:
            query_recent = "SELECT email_id, account_id, subject, from_, received_at FROM emails WHERE account_id = :account_id ORDER BY received_at DESC LIMIT 5"
            params_recent = {"account_id": account_id}
        else:
            query_recent = "SELECT email_id, account_id, subject, from_, received_at FROM emails ORDER BY received_at DESC LIMIT 5"
            params_recent = {}
        recent_emails_db = await database.fetch_all(query_recent, values=params_recent)
        
        return {
            "filter_account_id": account_id or "all_accounts",
            "database": {
                "total_emails": total_emails_in_db,
                "emails_with_embeddings": emails_with_embeddings,
                "recent_emails": [dict(row) for row in recent_emails_db]
            },
            "faiss": {
                "index_total": faiss_index.index.ntotal if hasattr(faiss_index, 'index') else "FAISS not initialized",
                "id_map_length": len(faiss_index.id_map) if hasattr(faiss_index, 'id_map') else "N/A",
                "sample_ids_from_map": faiss_index.id_map[:5] if hasattr(faiss_index, 'id_map') and faiss_index.id_map else []
            },
            "potential_issues": {
                "faiss_empty": not (hasattr(faiss_index, 'index') and faiss_index.index.ntotal > 0),
                "missing_embeddings_in_db": emails_with_embeddings < total_emails_in_db,
                "faiss_db_embedding_mismatch": hasattr(faiss_index, 'index') and faiss_index.index.ntotal != emails_with_embeddings if total_emails_in_db > 0 else "N/A (no emails in DB or embeddings count mismatch)"
            }
        }
    except Exception as e:
        print(f"Error in /debug/status: {str(e)}")
        return {"error": str(e)}


@app.post("/admin/rebuild-faiss-from-db")
async def rebuild_faiss_from_database_admin():
    """Rebuild FAISS index from all emails with embeddings in database."""
    try:
        # Get all emails with embeddings
        query = "SELECT email_id, embedding FROM emails WHERE embedding IS NOT NULL"
        result = await database.fetch_all(query)
        
        if not result:
            return {
                "success": False,
                "message": "No emails with embeddings found in database.",
                "action_needed": "Ensure emails are processed and embeddings are generated (e.g., via /emails/batch_add or a dedicated embedding generation endpoint)."
            }
        
        print(f"Found {len(result)} emails with embeddings in DB for FAISS rebuild.")
        
        # Clear existing FAISS index
        # Ensure faiss_index and its components are accessible and correctly typed
        if not hasattr(faiss_index, 'index') or not hasattr(faiss_index, 'id_map'):
             return {"success": False, "message": "FAISS index object not properly initialized."}

        # Reinitialize FAISS index (assuming DIM is defined, e.g., 1536 for OpenAI)
        DIM = 1536 # Or get from faiss_index.index.d if already initialized
        faiss_index.index = faiss.IndexFlatL2(DIM) 
        faiss_index.id_map = []
        
        email_ids_for_faiss = []
        embeddings_bytes_for_faiss = []
        
        valid_embeddings_count = 0
        invalid_embeddings_details = []
        
        for row in result:
            email_id = row['email_id']
            embedding_data = row['embedding']
            
            # Validate embedding data (e.g., correct byte length for 1536 float32)
            expected_byte_length = DIM * 4 # 1536 floats * 4 bytes/float
            if embedding_data and isinstance(embedding_data, bytes) and len(embedding_data) == expected_byte_length:
                email_ids_for_faiss.append(email_id)
                embeddings_bytes_for_faiss.append(embedding_data)
                valid_embeddings_count += 1
            else:
                invalid_embeddings_details.append({
                    "email_id": email_id, 
                    "embedding_present": bool(embedding_data),
                    "type": str(type(embedding_data)),
                    "length": len(embedding_data) if embedding_data else 0,
                    "expected_length": expected_byte_length
                })
        
        if not email_ids_for_faiss:
            return {
                "success": False,
                "message": "No valid embeddings found to add to FAISS index after filtering.",
                "total_from_db_with_embedding_column": len(result),
                "invalid_embedding_samples": invalid_embeddings_details[:5] # Show first 5 issues
            }
        
        # Add to FAISS index
        faiss_index.add(email_ids_for_faiss, embeddings_bytes_for_faiss)
        faiss_index.save() # Assuming a save method exists in your FaissIndex class
        
        return {
            "success": True,
            "message": "FAISS index rebuilt successfully.",
            "total_from_db_with_embedding_column": len(result),
            "valid_embeddings_added_to_faiss": valid_embeddings_count,
            "faiss_index_total_after_rebuild": faiss_index.index.ntotal,
            "faiss_id_map_length_after_rebuild": len(faiss_index.id_map),
            "invalid_embedding_details_count": len(invalid_embeddings_details),
            "invalid_embedding_samples": invalid_embeddings_details[:5]
        }
        
    except Exception as e:
        print(f"Error in /admin/rebuild-faiss-from-db: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": f"FAISS rebuild failed: {str(e)}"}