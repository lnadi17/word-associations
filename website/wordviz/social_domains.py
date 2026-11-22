"""Social domains and prime nodes from Table 2 of Abramski et al."""

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
