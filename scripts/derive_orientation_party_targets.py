"""Reproduce the website's orientation and US-party target lists from public data.

Run with Exp2Condensed.csv (https://osf.io/download/c3sv2/) and the Dataverse
TSV (https://dataverse.harvard.edu/api/access/datafile/7234329) as the two
positional arguments. To regenerate the public evidence CSVs from the repo root:

    python3 scripts/derive_orientation_party_targets.py Exp2Condensed.csv party.tab --evidence-dir website/public/assets/evidence

The JSON printed to stdout contains the two ordered target lists for each design.
No network access or model activations are required.
"""

import argparse
from collections import Counter, defaultdict
from contextlib import closing
import csv
import hashlib
import json
from pathlib import Path
import re
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
FILE_HASHES = (
    "4ae3cd0b466a99a2f362a5c808eaed7fd1f3f55394a1e189b84941adde14a18a",
    "d664d47539af6a8641b2432040fa51ffa1712a4640bb6a9d8866d009f2d76574",
)
NETWORKS = ("Humans", "Gpt5_mini", "Haiku", "Gemma", "Llama3", "Mistral")
ORIENTATION_PAIRS = (
    ("Gay men", "Straight men"),
    ("Lesbian women", "Straight women"),
    ("Bisexual men", "Straight men"),
    ("Bisexual women", "Straight women"),
    ("Bisexual men", "Gay men"),
    ("Bisexual women", "Lesbian women"),
)
ORIENTATION_LABELS = set("gay lesbian bisexual straight homosexual heterosexual".split())
PARTY_LABELS = set("democrat democrats democratic republican republicans gop".split())
WORD = re.compile(r"[a-z]+(?:-[a-z]+)*")


def common_vocabulary():
    vocabularies = []
    for name in NETWORKS:
        path = ROOT / "website/wordviz/resources/social" / f"{name}.sqlite"
        with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as db:
            vocabularies.append({node for node, in db.execute("SELECT node FROM activation")})
    return set.intersection(*vocabularies)


def ranked(first, second, first_total, second_total, minimum, eligible):
    # Integer numerators rank the exact difference in within-group proportions.
    ranked_terms = []
    for word, n in first.items():
        difference = n * second_total - second[word] * first_total
        if n >= minimum and difference > 0 and word in eligible:
            ranked_terms.append((-difference, -n, word))
    return [word for _, _, word in sorted(ranked_terms)[:25]]


def orientation_evidence(path, vocab):
    counts = defaultdict(Counter)
    source_codes = defaultdict(set)
    totals = Counter()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            group = row["TargetGroup"]
            totals[group] += 1
            # Keep a single original label for synonymous single-word alternatives.
            # Exclude phrases, including negations, rather than extract fragments.
            labels = [word.strip() for word in row["Final"].strip().casefold().split("/")]
            if all(WORD.fullmatch(word) for word in labels):
                counts[group][labels[0]] += 1
                source_codes[labels[0]].add(row["Final"].strip())
    eligible = vocab - ORIENTATION_LABELS
    rows = []
    for a, b in ORIENTATION_PAIRS:
        for group, other in ((a, b), (b, a)):
            words = ranked(counts[group], counts[other], totals[group], totals[other], 2, eligible)
            for rank, word in enumerate(words, 1):
                n, m = counts[group][word], counts[other][word]
                rows.append({
                    "design": f"{a} / {b}", "target_group": group, "comparison_group": other,
                    "target_prime": group.split()[0].lower(), "comparison_prime": other.split()[0].lower(),
                    "rank": rank, "word": word, "source_codes": " | ".join(sorted(source_codes[word])),
                    "target_nominations": n, "target_total_nominations": totals[group],
                    "comparison_nominations": m, "comparison_total_nominations": totals[other],
                    "selection_score_nomination_share_difference": f"{n / totals[group] - m / totals[other]:.9f}",
                })
    return rows


def party_evidence(path, vocab):
    # Follow the authors' Appendix A recodes and count a word once per respondent per party.
    recodes = {"wealth": "rich", "hard working": "hardworking", "nothing": "[nonresponse]",
               "open minded": "open-minded", "closed minded": "closed-minded"}
    responses = defaultdict(set)
    source_codes = defaultdict(set)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            for party in ("D", "R"):
                original = row[f"Gen{party}_recode"][:30].strip().casefold()
                word = recodes.get(original, original)
                if word and word != "[nonresponse]":
                    responses[row["sample"], row["ID"], party].add(word)
                    source_codes[word].add(original)
    # Paired respondents must have supplied at least one response to both prompts.
    counts = defaultdict(Counter)
    denominators = Counter()
    for (sample, respondent, party), words in responses.items():
        other = "R" if party == "D" else "D"
        if (sample, respondent, other) in responses:
            counts[sample, party].update(words)
            denominators[sample, party] += 1
    rows = []
    labels = {"D": ("Democratic supporters", "democrat"), "R": ("Republican supporters", "republican")}
    for party, other in (("D", "R"), ("R", "D")):
        eligible = {word for word in vocab - PARTY_LABELS if WORD.fullmatch(word) and all(
            counts[wave, party][word] > counts[wave, other][word] for wave in ("1", "2")
        )}
        # The respondent denominators are identical for the two paired prompts.
        totals = counts["1", party] + counts["2", party]
        others = counts["1", other] + counts["2", other]
        for rank, word in enumerate(ranked(totals, others, 1, 1, 10, eligible), 1):
            row = {
                "design": "Democratic supporters / Republican supporters",
                "target_group": labels[party][0], "comparison_group": labels[other][0],
                "target_prime": labels[party][1], "comparison_prime": labels[other][1],
                "rank": rank, "word": word, "source_codes": " | ".join(sorted(source_codes[word])),
                "target_nominators": totals[word], "comparison_nominators": others[word],
                "pooled_matched_respondents": denominators["1", party] + denominators["2", party],
                "selection_score_nominator_difference": totals[word] - others[word],
            }
            for wave, year in (("1", "2016"), ("2", "2021")):
                row[f"target_nominators_{year}"] = counts[wave, party][word]
                row[f"comparison_nominators_{year}"] = counts[wave, other][word]
                row[f"matched_respondents_{year}"] = denominators[wave, party]
            rows.append(row)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("orientation_csv", type=Path)
    parser.add_argument("party_tsv", type=Path)
    parser.add_argument("--evidence-dir", type=Path, help="Directory for the two ranked evidence CSVs")
    args = parser.parse_args()
    for path, expected in zip((args.orientation_csv, args.party_tsv), FILE_HASHES):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            parser.error(f"{path.name} differs from the cited version")
    vocab = common_vocabulary()
    results = defaultdict(dict)
    for filename, rows in (("orientation-targets.csv", orientation_evidence(args.orientation_csv, vocab)),
                           ("party-targets.csv", party_evidence(args.party_tsv, vocab))):
        for row in rows:
            results[row["design"]].setdefault(row["target_group"], []).append(row["word"])
        if args.evidence_dir:
            args.evidence_dir.mkdir(parents=True, exist_ok=True)
            with (args.evidence_dir / filename).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
    print(json.dumps({name: list(groups.values()) for name, groups in results.items()}, ensure_ascii=False, indent=2))
