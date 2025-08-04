# utils.py

import re

def clean_text(text):
    """
    Basic text cleaning: remove extra spaces, line breaks, and non-textual elements
    """
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text
