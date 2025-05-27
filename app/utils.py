from dateutil import parser
from datetime import datetime, timedelta

def parse_timeframe(text: str):
    # Very simple heuristic: detect "next X days"
    import re
    match = re.search(r'next (\d+) days', text.lower())
    if match:
        days = int(match.group(1))
        start = datetime.utcnow()
        end = start + timedelta(days=days)
        return start, end
    return None, None
