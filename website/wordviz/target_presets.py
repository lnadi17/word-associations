"""Published stereotype stimuli and corpus associations used as editable targets."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Study:
    citation: str
    url: str
    context: str


@dataclass(frozen=True)
class TargetPreset:
    study: Study
    words: tuple[str, ...]


_RELIGIOUS_TRAITS = Study(
    "Erentzen et al. (2023), Study 3",
    "https://doi.org/10.1177/13684302221138036",
    "US participants rated perceived traits of Christians and Muslims. Applying these "
    "scales to other religion primes extends the original comparison.",
)
_RELIGIOUS_WARMTH = Study(
    "Ginguené et al. (2026), Study 1",
    "https://doi.org/10.1017/S1755048326100601",
    "French participants rated warmth for Muslims, Jews, Catholics and Protestants, "
    "with pooled categories for Buddhists and Hindus, and for atheists and agnostics.",
)
_ETHNIC_COMPETENCE = Study(
    "Fiske et al. (2002), Study 2, Table 7",
    "https://doi.org/10.1037/0022-3514.82.6.878",
    "US participants rated how society viewed social groups, including Asian, "
    "Black, Hispanic and White groups. These six items measured perceived competence.",
)
_HISTORICAL_TRAITS = Study(
    "Charlesworth et al. (2022), Table 1",
    "https://doi.org/10.1073/pnas.2121798119",
    "In English-language Google Books from 1800–1999, these traits were more closely "
    "associated with the first group in the preset label. The labels preserve the "
    "study’s categories. You can test these words with any network primes.",
)
_ORIENTATION_TRAITS = Study(
    "McGorray & Petsko (2024), Experiment 2, Tables 5–6",
    "https://doi.org/10.1177/19485506231183467",
    "US participants named cultural stereotypes for men and women of different "
    "sexual orientations. The straight and bisexual sets draw from both groups of targets.",
)
_PARTY_STEREOTYPES = Study(
    "Rothschild et al. (2019), Table 1",
    "https://doi.org/10.1007/s11109-018-9457-5",
    "US survey respondents described supporters of each party. These are frequently "
    "reported associations across the pooled samples, including traits and demographic terms.",
)


TARGET_PRESETS = {
    "religion": {
        "Perceived religiousness": TargetPreset(
            _RELIGIOUS_TRAITS, ("religious", "faithful", "holy", "pure", "moral", "modest"),
        ),
        "Perceived aggression": TargetPreset(
            _RELIGIOUS_TRAITS, ("aggressive", "angry", "violent", "dangerous", "mean"),
        ),
        "Perceived warmth": TargetPreset(
            _RELIGIOUS_WARMTH, ("benevolent", "altruistic", "honest", "sincere", "sociable", "warm"),
        ),
    },
    "ethnicity": {
        "Perceived competence": TargetPreset(
            _ETHNIC_COMPETENCE,
            ("competent", "confident", "capable", "efficient", "intelligent", "skillful"),
        ),
        # First eight traits from each published ranking found in all six networks.
        "Asian vs. White · historical text": TargetPreset(
            _HISTORICAL_TRAITS,
            ("pompous", "verbal", "curious", "traditional", "melancholy", "solemn", "artificial", "sensual"),
        ),
        "Hispanic vs. White · historical text": TargetPreset(
            _HISTORICAL_TRAITS,
            ("verbal", "pompous", "formal", "solemn", "abrupt", "diplomatic", "traditional", "evasive"),
        ),
        "Native American vs. White · historical text": TargetPreset(
            _HISTORICAL_TRAITS,
            ("rude", "earthy", "wholesome", "spontaneous", "artificial", "kind", "sensual", "lonely"),
        ),
        "Black vs. White · historical text": TargetPreset(
            _HISTORICAL_TRAITS,
            ("earthy", "lonely", "cruel", "sensual", "lifeless", "deceitful", "helpless", "rebellious"),
        ),
        "White vs. Black · historical text": TargetPreset(
            _HISTORICAL_TRAITS,
            ("critical", "polite", "hostile", "decisive", "friendly", "diplomatic", "understanding", "able"),
        ),
    },
    "orientation": {
        "Reported stereotypes of straight people": TargetPreset(
            _ORIENTATION_TRAITS,
            ("masculine", "strong", "aggressive", "sexual", "feminine", "kind", "domestic", "emotional"),
        ),
        "Reported stereotypes of gay men": TargetPreset(
            _ORIENTATION_TRAITS, ("feminine", "fashionable", "flamboyant", "promiscuous", "loud", "weak"),
        ),
        "Reported stereotypes of lesbian women": TargetPreset(
            _ORIENTATION_TRAITS, ("masculine", "mean", "fat", "athletic", "feminist", "unattractive"),
        ),
        "Reported stereotypes of bisexual people": TargetPreset(
            _ORIENTATION_TRAITS, ("feminine", "promiscuous", "confused", "indecisive", "masculine", "sexual"),
        ),
    },
    "politics": {
        "Democratic supporter associations": TargetPreset(
            _PARTY_STEREOTYPES,
            ("liberal", "open-minded", "caring", "youth", "smart", "poor", "equality", "educated", "ignorant"),
        ),
        "Republican supporter associations": TargetPreset(
            _PARTY_STEREOTYPES, ("conservative", "rich", "white", "ignorant", "religious", "tradition"),
        ),
    },
}
