"""Prepare browser assets and verify the server-side research resources."""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil

import plotly

from wordviz.bias_norms import valence_result, emotion_result
from wordviz.experiment_data import metadata
from wordviz.explore_archives import BUNDLED, BUNDLE_DIR, DATA_DIR, MANIFEST
from wordviz.explore_data import vocabulary
from wordviz.models import GRAPH_MODELS, MATRIX_MODELS
from wordviz.norm_lexicons import SERVER_PACK, NormSourceError, load_lexicons, prepare_pack
from wordviz.path_data import load_graph
from wordviz.social_data import domain_info
from wordviz.social_domains import DOMAIN_PRIMES


def write_remote_archives():
    release = Path(__file__).with_name("release-files.json")
    commit = json.loads(release.read_text())["commit"] if release.is_file() else os.environ.get("VERCEL_GIT_COMMIT_SHA")
    if not commit or not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ValueError("Packaging remote Explore files requires the source Git commit.")
    entries = {}
    BUNDLE_DIR.mkdir(exist_ok=True)
    for model in GRAPH_MODELS:
        archive = DATA_DIR / f"{model}.zip"
        if model in BUNDLED:
            shutil.copyfile(archive, BUNDLE_DIR / archive.name)
            continue
        with archive.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        entries[model] = {"sha256": digest, "bytes": archive.stat().st_size}
    MANIFEST.write_text(json.dumps({"commit": commit, "archives": entries}, separators=(",", ":")))
    print(f"Prepared {len(entries)} commit-pinned Explore archives", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vad", type=Path, help="Official NRC VAD v1 ZIP for an offline build")
    parser.add_argument("--emolex", type=Path, help="Official NRC EmoLex v0.92 ZIP for an offline build")
    parser.add_argument("--package-explore", action="store_true", help="Record verified archives for the Vercel function.")
    args = parser.parse_args()
    if bool(args.vad) != bool(args.emolex):
        parser.error("Supply both NRC archives for an offline build.")
    try:
        if args.vad or not SERVER_PACK.is_file():
            archives = {"vad": args.vad, "emotion": args.emolex} if args.vad else None
            prepare_pack(SERVER_PACK, archives)
        load_lexicons()
    except NormSourceError as error:
        if args.vad:
            raise
        print(f"{error}. Custom norm analyses will retry the verified download on demand.", flush=True)
    for model in GRAPH_MODELS:
        if not vocabulary(model) or not load_graph(model)["words"]:
            raise SystemExit(f"Empty serving data for {model}")
    if not set(MATRIX_MODELS).issubset(metadata("ldt")["activation"]):
        raise SystemExit("Incomplete LDT matrix snapshot")
    for model in MATRIX_MODELS:
        for domain in DOMAIN_PRIMES:
            primes = domain_info(domain, model)["primes"]
            if set(primes) != {row["prime"] for row in valence_result(domain, model)["rows"]}:
                raise SystemExit(f"Incomplete valence data for {model}/{domain}")
            if len(primes) >= 2 and len(emotion_result(domain, model, *primes[:2])["rows"]) != 8:
                raise SystemExit(f"Incomplete emotion data for {model}/{domain}")
    source = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    target = Path(__file__).parent / "public" / "vendor" / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if args.package_explore:
        write_remote_archives()
