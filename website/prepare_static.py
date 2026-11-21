"""Put the Plotly browser bundle on Vercel's static CDN during the build."""

from pathlib import Path
import shutil

import plotly

from wordviz.experiment_data import metadata
from wordviz.explore_data import vocabulary
from wordviz.models import GRAPH_MODELS, MATRIX_MODELS
from wordviz.path_data import load_graph


if __name__ == "__main__":
    for model in GRAPH_MODELS:
        if not vocabulary(model) or not load_graph(model)["words"]:
            raise SystemExit(f"Empty serving data for {model}")
    if set(metadata("ldt")["activation"]) != set(MATRIX_MODELS):
        raise SystemExit("Incomplete LDT matrix snapshot")
    source = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    target = Path(__file__).parent / "public" / "vendor" / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
