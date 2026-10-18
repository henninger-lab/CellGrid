"""Normalize directory paths pasted into text controls."""


def normalize_directory_path(value):
    """Remove surrounding whitespace and one matching pair of quote marks."""
    value = str(value).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value
