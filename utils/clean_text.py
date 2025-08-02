import re

def clean_text(text: str, chars_to_ignore: str = r'[,?.!\-;:"“%\'�]') -> str:
    """
    Clean a transcript string by removing special characters and lowering the case.

    Args:
        text (str): The input transcript.
        chars_to_ignore (str): Regex pattern of characters to remove.

    Returns:
        str: Cleaned transcript.
    """
    cleaned = re.sub(chars_to_ignore, '', text).lower()
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned