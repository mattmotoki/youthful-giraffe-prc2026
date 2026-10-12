"""Small, testable helpers used by released lineage evidence."""
import re


def profile_is_configured(profile: str | None, credentials_text: str) -> bool:
    """Return true only when a non-empty selected profile occurs in credentials."""
    if not profile:
        return False
    sections = re.findall(r"(?m)^\s*\[([^]\r\n]+)\]\s*$", credentials_text)
    return profile in sections
