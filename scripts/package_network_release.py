"""Package the finished networks from the statistical analysis for a GitHub data release."""

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[1]
ANALYSES = ROOT / "data/LDT_analyses"
MANIFEST = ROOT / "website/wordviz/resources/network_release.json"


def package(model, directory, external_trials):
    key = model.lower()
    trials = ROOT / f"data/processed/{key}_processed.csv"
    members = [
        trials if trials.is_file() else external_trials[key],
        ROOT / f"data/graphs/edge_lists/FA_{key}_edgelist.csv",
        ROOT / f"data/graphs/igraphs/FA_{key}_ig.graphml",
        ANALYSES / f"FA_matrices/{model}_LDT.csv",
        ANALYSES / f"FA_matrices/{model}_gender.csv",
    ]
    audit = ROOT / f"data/summary_tables/collection_audits/{key}.json"
    if audit.is_file():
        members.append(audit)
    destination = directory / f"{key}.zip"
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        for member in members:
            data = member.read_bytes()
            if not data or data.startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError(f"Missing materialized research data in {member}")
            entry = ZipInfo(member.relative_to(ROOT).as_posix())
            archive.writestr(entry, data, compress_type=ZIP_DEFLATED, compresslevel=6)
    with destination.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return key, {"sha256": digest, "bytes": destination.stat().st_size}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9._-]+", args.tag):
        parser.error("Expected a GitHub release tag")
    args.output.mkdir(parents=True, exist_ok=True)
    models = json.loads((ANALYSES / "Statistical_tests/all_networks/coverage.json").read_text())["models"]
    external = {p.stem[3:].lower(): p for p in (ROOT / "lexicon/LWOW_datasets/processed_datasets").glob("FA_*.csv")}
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = dict(pool.map(lambda model: package(model, args.output, external), models))
    MANIFEST.write_text(json.dumps({"tag": args.tag, "networks": records}, indent=2) + "\n")
    (args.output / "SHA256SUMS").write_text("".join(
        f"{record['sha256']}  {key}.zip\n" for key, record in records.items()
    ))
