"""Reproduce the website's ethnic and religious target lists from participant responses.

Pass the three original CSVs from Nicolas, Bai & Fiske (2022), in study order:
https://osf.io/download/ef3j7/?version=2
https://osf.io/download/pjkde/?version=1
https://osf.io/download/vkpy5/?version=1

Prints each word, the nomination counts on both sides, and the selection score.
Use --check to verify the website's lists and order. No graph data is used.

Regenerate the public aggregate CSV from the repository root:
  python3 scripts/derive_identity_targets.py Dictionary_1.csv "Dictionary 2.csv" "Dictionary 3.csv" \
      --check --output website/public/assets/evidence/identity-targets.csv

In that CSV, a blank study column means the study was outside that design.
Counts are distinct respondents who named a word, denominators are all
respondents assigned that group. The score is the mean across included studies
of (assigned-side count / denominator) minus (other-side count / denominator).
"""

import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import sys

SOURCES = (
    # File checksum, group slots per participant, responses per group.
    ("1d9022220aa5c10c425a0335a6a5326f8fe2893d2226df3c668d903f8c962b2d", 6, 6),
    ("ea77b9999b8284798b7be07947ce5d7702639edc5dc22fec296af7457cd537b3", 30, 2),
    ("0a8f2c1620d28fafe2fa034cb389bb7c55414f42c50537eac5aa2f25097ae933", 6, 3),
)

# Explicit spelling and singular/plural normalizations used across all studies.
# No synonyms, imputations, graph-dependent selection or splitting of phrases.
VARIANTS = {
    "aggresive": "aggressive", "judgemental": "judgmental", "religous": "religious",
    "close-minded": "closed-minded", "non-believers": "nonbeliever",
    "nonbelievers": "nonbeliever", "non-believer": "nonbeliever",
    "arabs": "arab", "bankers": "banker", "dancers": "dancer", "followers": "follower",
    "idiots": "idiot", "immigrants": "immigrant", "indians": "indian", "killers": "killer",
    "monks": "monk", "scientists": "scientist", "terrorists": "terrorist",
    "thinkers": "thinker", "thugs": "thug",
}
GROUPS = {
    "Christian": ("christian", "christians"), "Muslim": ("muslim", "muslims"),
    "Atheists": ("atheist", "atheists"), "Jewish": ("jewish", "jew", "jews"),
    "Buddhists": ("buddhist", "buddhists"), "Black": ("black", "blacks"),
    "White": ("white", "whites"), "Asian": ("asian", "asians"),
    "Hispanic": ("hispanic", "hispanics"),
}
PAIRS = {
    "religion": (("Christian", "Muslim"), ("Christian", "Atheists"),
                 ("Christian", "Jewish"), ("Christian", "Buddhists")),
    "ethnicity": (("Black", "White"), ("Asian", "White"), ("Hispanic", "White")),
}


def read_study(path, source):
    checksum, slots, responses = source
    if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
        raise ValueError(f"Unexpected file version: {path}")
    counts = defaultdict(Counter)
    assigned = Counter()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            for slot in range(1, slots + 1):
                group = row[f"G_{slot}"]
                if group not in GROUPS:
                    continue
                assigned[group] += 1
                words = set()
                for response in range(1, responses + 1):
                    raw = row[f"t.{response}_{slot}"].strip().casefold()
                    word = VARIANTS.get(raw, raw)
                    if re.fullmatch(r"[a-z]+(?:-[a-z]+)*", word) and word not in {"na", "none", "unknown"}:
                        words.add(word)
                counts[group].update(words)
    return counts, assigned


def rank(studies, first, second):
    excluded = set(GROUPS[first] + GROUPS[second])
    vocabulary = set().union(*(counts[first] for counts, _ in studies)) - excluded
    ranked = []
    for word in vocabulary:
        first_count = sum(counts[first][word] for counts, _ in studies)
        if first_count < 2:
            continue
        difference = sum(
            Fraction(counts[first][word], assigned[first]) - Fraction(counts[second][word], assigned[second])
            for counts, assigned in studies
        ) / len(studies)
        if difference > 0:
            other_count = sum(counts[second][word] for counts, _ in studies)
            ranked.append((difference, first_count, word, other_count))
    ranked.sort(key=lambda x: (-x[0], -x[1], x[2]))
    return [{"word": word, "assigned_side_count": own, "other_side_count": other,
             "share_difference": float(score)} for score, own, word, other in ranked[:25]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs=3)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, help="Write aggregate counts for every selected target as CSV")
    args = parser.parse_args()
    studies = [read_study(path, source) for path, source in zip(args.csv, SOURCES)]
    if args.check:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "website"))
        from wordviz.research_identity_targets import STUDIES
    report = {}
    audit_rows = []
    for domain, pairs in PAIRS.items():
        study_ids = (1,) if domain == "ethnicity" else (2, 3)
        sample = [studies[i - 1] for i in study_ids]
        for first, second in pairs:
            sides = [rank(sample, first, second), rank(sample, second, first)]
            primes = (GROUPS[first][0], GROUPS[second][0])
            if args.check:
                preset = next(x for x in STUDIES[domain].values() if x.primes == primes)
                words = tuple(tuple(row["word"] for row in side) for side in sides)
                if words != preset.targets:
                    raise SystemExit(f"Stored target mismatch for {primes}")
            report[" vs. ".join(primes)] = {
                "raters_by_study": [{group: n[group] for group in (first, second)} for _, n in sample],
                "sides": sides,
            }
            if args.output:
                for (own, other), rows in zip(((first, second), (second, first)), sides):
                    for position, row in enumerate(rows, 1):
                        word = row["word"]
                        values = {"domain": domain, "prime_1": primes[0], "prime_2": primes[1],
                                  "assigned_group": own, "other_group": other, "word": word,
                                  "rank": position, "mean_share_difference": row["share_difference"]}
                        for study_id, (counts, denominators) in zip(study_ids, sample):
                            values[f"study_{study_id}_assigned_count"] = counts[own][word]
                            values[f"study_{study_id}_assigned_denominator"] = denominators[own]
                            values[f"study_{study_id}_other_count"] = counts[other][word]
                            values[f"study_{study_id}_other_denominator"] = denominators[other]
                        audit_rows.append(values)
    if args.output:
        columns = ["domain", "prime_1", "prime_2", "assigned_group", "other_group", "word", "rank",
                   "mean_share_difference"]
        columns.extend(f"study_{i}_{side}_{measure}" for i in (1, 2, 3)
                       for side in ("assigned", "other") for measure in ("count", "denominator"))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(audit_rows)
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
