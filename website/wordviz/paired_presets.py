"""Resolve study designs and user-assigned prime/target lists for the analysis views."""

from functools import lru_cache
from itertools import combinations

from .experiment_data import metadata
from .research_design import PairedStudy
from .research_identity_targets import STUDIES as IDENTITY_STUDIES
from .research_other_targets import STUDIES as OTHER_STUDIES


@lru_cache(maxsize=1)
def study_catalog():
    female, male = metadata("targets")
    gender = PairedStudy(
        primes=("woman", "man"), labels=("Female-related traits", "Male-related traits"),
        targets=(tuple(female), tuple(male)), citation="Abramski et al., Tables 2–3 and Figure 3",
        url="https://arxiv.org/html/2510.24488#S2",
        context="The study used 25 traits per side and five prime pairs. This template tests one of those pairs.",
    )
    return {"gender": {"Gender traits · woman and man": gender}, **IDENTITY_STUDIES, **OTHER_STUDIES}


def templates():
    return {f"{domain}/{title}": study for domain, entries in study_catalog().items()
            for title, study in entries.items()}


def _as_preset(study):
    return {"primes": study.primes, "labels": study.labels, "groups": study.targets,
            "citation": study.citation, "url": study.url, "context": study.context,
            "data_url": study.data_url, "evidence_url": study.evidence_url,
            "survey_nominations": study.survey_nominations}


def paired_presets(domain):
    if isinstance(domain, str):
        return {name: _as_preset(study) for name, study in study_catalog().get(domain, {}).items()}
    if not isinstance(domain, dict):
        return {}
    # Only explicitly assigned lists imply a directional hypothesis. Legacy lists
    # keep their original exploratory use until the researcher assigns their primes.
    options = {}
    assignments = domain.get("list_primes", {})
    source = templates().get(domain.get("study_template"))
    for first, second in combinations(domain["lists"], 2):
        left, right = assignments.get(first), assignments.get(second)
        if not left or not right or left == right:
            continue
        groups = (domain["lists"][first], domain["lists"][second])
        title = f"{first} / {second} · {left} vs {right}"
        entry = {"primes": (left, right), "labels": (first, second), "groups": groups,
                 "citation": "", "url": "", "context": "You assigned each target list to its expected prime.",
                 "data_url": ""}
        if source and {left, right} == set(source.primes):
            original = tuple(source.targets[source.primes.index(prime)] for prime in (left, right))
            exact = all(set(supplied) == set(cited) for supplied, cited in zip(groups, original))
            context = source.context
            if (left, right) != source.primes:
                context = f"Original study order: {source.primes[0]}, then {source.primes[1]}. {context}"
            if not exact:
                context = "Edited target sets. Starting study: " + context
            entry.update(citation=source.citation, url=source.url, data_url=source.data_url,
                         evidence_url=source.evidence_url, context=context)
            source_labels = tuple(source.labels[source.primes.index(p)] for p in (left, right))
            entry["survey_nominations"] = source.survey_nominations and exact and (first, second) == source_labels
        options[title] = entry
    for designs in study_catalog().values():
        for title, study in designs.items():
            if set(study.primes).issubset(domain["primes"]):
                if any(row["primes"] == study.primes and tuple(map(tuple, row["groups"])) == study.targets
                       for row in options.values()):
                    continue
                options.setdefault(title, _as_preset(study))
    if not options:
        left, right = domain["primes"][:2]
        options["Enter two target sets"] = {
            "primes": (left, right), "labels": (f"Targets for {left}", f"Targets for {right}"),
            "groups": ((), ()), "manual_targets": True, "citation": "", "url": "", "data_url": "",
            "context": "No paired study in this catalog matches these primes. Enter the targets you expect "
                       "each prime to activate more strongly. To save lists or change their expected primes, "
                       "choose Edit under Social domain.",
        }
    return options
