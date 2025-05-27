from datetime import datetime, timedelta
from app.db import get_emails_in_date_range

def is_important(email_doc):
    keywords = ['urgent', 'action required', 'meeting', 'interview']
    text = (email_doc['subject'] + ' ' + email_doc['body']).lower()
    return any(kw in text for kw in keywords)

async def get_important_emails():
    today = datetime.utcnow()
    week_ago = today - timedelta(days=7)
    emails = await get_emails_in_date_range(week_ago, today)
    important = [e for e in emails if is_important(e)]
    return important