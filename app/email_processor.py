import numpy as np
from app.openai_client import get_embedding # Assuming this exists in your app structure
from datetime import datetime
from dateutil.parser import parse as parse_date

def embedding_to_bytes(embedding_list):
    return np.array(embedding_list, dtype=np.float32).tobytes()

def bytes_to_embedding(b):
    return np.frombuffer(b, dtype=np.float32)

def extract_metadata(email: dict, account_id: str): # Added account_id parameter
    subject = email.get('subject', '')
    from_ = email.get('from_') or email.get('from', '')
    body = email.get('body') or email.get('snippet', '')
    
    try:
        received_at_str = email.get('date')
        if received_at_str:
            # Attempt to parse, handling potential timezone offsets like 'Z'
            if isinstance(received_at_str, str) and received_at_str.endswith('Z'):
                 received_at_str = received_at_str[:-1] + '+00:00'
            received_at = parse_date(received_at_str)
        else:
            received_at = datetime.utcnow()
    except Exception as e:
        print(f"Date parsing error for email {email.get('id', 'unknown')}: {e}. Raw date: {email.get('date')}")
        received_at = datetime.utcnow() # Fallback to current UTC time
        
    category = "general"
    content_to_check = (str(subject) + str(body)).lower() # Ensure subject and body are strings
    if any(kw in content_to_check for kw in ['meeting', 'interview', 'call']):
        category = "meeting"
    elif any(kw in content_to_check for kw in ['urgent', 'asap', 'important']):
        category = "urgent"
    
    return {
        "email_id": email['id'],
        "account_id": account_id, # Included account_id
        "subject": subject,
        "from_": from_,
        "body": body,
        "received_at": received_at,
        "category": category
    }

async def process_email(email_data: dict): # Changed parameter name for clarity
    """
    Processes a single email dictionary.
    The input email_data dictionary MUST contain 'account_id'.
    """
    try:
        email_id = email_data.get('id', 'unknown_id')
        account_id = email_data.get('account_id')

        if not account_id:
            print(f"Error processing email {email_id}: 'account_id' is missing.")
            # Optionally, raise an error or return a specific structure indicating failure
            # For now, we'll try to proceed but log a warning.
            # It's better to enforce this at the API level.
            # raise ValueError(f"account_id is missing for email {email_id}")
            account_id = "unknown_account" # Fallback, but not ideal

        print(f"Processing email: {email_id} for account: {account_id}")
        
        # Pass account_id to extract_metadata
        metadata = extract_metadata(email_data, account_id)
        
        full_text = f"Subject: {metadata.get('subject', '')}\nFrom: {metadata.get('from_', '')}\nAccount: {metadata.get('account_id', '')}\n\n{metadata.get('body', '')}"
        
        # Get embedding
        embedding = await get_embedding(full_text) # Ensure get_embedding can handle the text
        metadata['embedding'] = embedding_to_bytes(embedding)
        metadata['full_text'] = full_text
        
        print(f"Processed email metadata for {metadata['email_id']} (Account: {metadata['account_id']})")
        return metadata
    except Exception as e:
        email_id_for_error = email_data.get('id', 'unknown_id')
        print(f"Error processing email {email_id_for_error}: {str(e)}")
        # Consider re-raising the exception or returning an error structure
        # For example, to allow the calling function to handle it:
        # raise ProcessEmailError(f"Failed to process email {email_id_for_error}: {e}") from e
        raise # Re-raise the exception to be caught by the caller
