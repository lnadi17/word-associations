"""Social domains and prime nodes from Table 2 of Abramski et al."""

import hashlib
import json

DOMAIN_LABELS = {
    "gender": "Gender",
    "religion": "Religion",
    "ethnicity": "Ethnicity",
    "orientation": "Sexual orientation",
    "politics": "Political party (US)",
}

DOMAIN_PRIMES = {
    "gender": ("woman", "man", "girl", "boy", "mother", "father", "female", "male",
               "feminine", "masculine"),
    # Table 2 prints "athiest". The graph node is "atheist" in all ten networks.
    "religion": ("christian", "muslim", "buddhist", "jewish", "atheist"),
    "ethnicity": ("european", "african", "hispanic", "asian", "indigenous"),
    "orientation": ("straight", "gay", "lesbian", "bisexual"),
    "politics": ("democrat", "republican"),
}


def resolve_domain(value):
    """Validate a panel's browser-supplied domain before using its contents."""
    if isinstance(value, str):
        return value if value in DOMAIN_LABELS else None
    from .custom_domains import validate_domain
    return validate_domain(value)


def primes_for(domain):
    return domain["primes"] if isinstance(domain, dict) else DOMAIN_PRIMES[domain]


def label_for(domain):
    return domain["name"] if isinstance(domain, dict) else DOMAIN_LABELS[domain]


def key_for(domain):
    if isinstance(domain, str):
        return domain
    # Editing a saved definition must invalidate old target and prime selections.
    definition = json.dumps(domain, ensure_ascii=False, separators=(",", ":")).encode()
    return domain["id"] + "-" + hashlib.sha256(definition).hexdigest()[:16]
