"""
Defines users, roles, and what departments each role can access.
"""

# Maps role -> list of departments that role can read
ROLE_ACCESS = {
    "hr":        ["hr", "general"],
    "finance":   ["finance", "general"],
    "marketing": ["marketing", "general"],
    "engineering": ["engineering", "general"],
    "ceo":       ["hr", "finance", "marketing", "engineering", "general"],
}

# Demo users — in a real app this would come from a database / auth provider
USERS = {
    "alice": {"role": "hr",          "display_name": "Alice (HR)"},
    "bob":   {"role": "finance",     "display_name": "Bob (Finance)"},
    "carol": {"role": "marketing",   "display_name": "Carol (Marketing)"},
    "dave":  {"role": "engineering", "display_name": "Dave (Engineering)"},
    "eve":   {"role": "ceo",         "display_name": "Eve (CEO)"},
}


def get_allowed_departments(role: str) -> list[str]:
    return ROLE_ACCESS.get(role, [])
