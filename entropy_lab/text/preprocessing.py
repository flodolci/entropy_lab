import re
import unicodedata
from typing import List

def preprocess_text(text: str) -> List[str]:
    """
    Tokenizer for stylometric / entropy analysis.

    - Unicode normalization (curly quotes -> regular forms, etc.)
    - Lowercase
    - Replace numbers with <num> (including decimals / percentages)
    - Keep words (including apostrophes and hyphens)
    - Keep punctuation as separate tokens (useful for style)
    - Keeps accented characters (French/German/Italian friendly)

    Examples:
    "Don't pay 12.5% now!" -> ["don't", "pay", "<num>", "%", "now", "!"]
    "state-of-the-art"      -> ["state-of-the-art"]
    """
    # Normalize Unicode (helps standardize quotes/dashes)
    text = unicodedata.normalize("NFKC", text)

    # Lowercase
    text = text.lower()

    # Standardize some common punctuation variants
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("—", "-").replace("–", "-")

    # Replace numbers (integers, decimals, commas) with <num>
    # Examples: 12, 12.5, 1,000, 3.1415
    text = re.sub(r"\b\d+(?:[.,]\d+)*\b", " <num> ", text)

    # Token pattern:
    # - <num>
    # - words with internal apostrophes/hyphens (e.g., don't, state-of-the-art)
    # - punctuation as separate tokens
    token_pattern = r"""
        <num>
        |
        [^\W\d_]+(?:[-'][^\W\d_]+)*   # Unicode letters, optional internal - or '
        |
        [.,!?;:%()"'/\-]              # punctuation tokens kept separately
    """

    tokens = re.findall(token_pattern, text, flags=re.VERBOSE | re.UNICODE)
    return tokens