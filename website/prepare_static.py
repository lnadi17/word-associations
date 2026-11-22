"""Prepare browser assets and verify the server-side research resources."""

import argparse
from pathlib import Path
import shutil

import plotly

from wordviz.bias_norms import valence_result, emotion_result
from wordviz.experiment_data import metadata
from wordviz.explore_data import vocabulary
from wordviz.models import GRAPH_MODELS, MATRIX_MODELS
from wordviz.norm_lexicons import SERVER_PACK, NormSourceError, load_lexicons, prepare_pack
from wordviz.path_data import load_graph
from wordviz.social_data import domain_info
from wordviz.social_domains import DOMAIN_PRIMES


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vad", type=Path, help="Official NRC VAD v1 ZIP for an offline build")
    parser.add_argument("--emolex", type=Path, help="Official NRC EmoLex v0.92 ZIP for an offline build")
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
    if set(metadata("ldt")["activation"]) != set(MATRIX_MODELS):
        raise SystemExit("Incomplete LDT matrix snapshot")
    for model in MATRIX_MODELS:
        for domain in DOMAIN_PRIMES:
            primes = domain_info(domain, model)["primes"]
            if len(primes) < 2 or not valence_result(domain, model)["rows"]:
                raise SystemExit(f"Incomplete valence data for {model}/{domain}")
            if len(emotion_result(domain, model, *primes[:2])["rows"]) != 8:
                raise SystemExit(f"Incomplete emotion data for {model}/{domain}")
    source = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    target = Path(__file__).parent / "public" / "vendor" / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
