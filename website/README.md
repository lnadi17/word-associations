# Word Associations

A Python Dash application for exploring the project's cleaned trials, saved activation matrices, and filtered graphs. The three pages are available at `/`, `/lexical-decision`, and `/implicit-bias`.

## Run locally

From the `website/` directory, use Python 3.13 and run

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

Open [the local explorer](http://127.0.0.1:8050/). Set `PORT` to choose a different port.

## Vercel deployment

Use **website** as the Vercel project’s Root Directory for a Git deployment, or submit the contents of this directory as the deployment root. The Flask entry point is `app.py:app`. The build installs the web dependencies and runs `python prepare_static.py` to verify the research stores and refresh the pinned Plotly browser bundle in `public/vendor/`. Include the static assets and `wordviz/resources/` in the deployment source. Vercel serves the JavaScript and stylesheet from its CDN, while the function reads the research data. The application requires no credentials, external database, or model API.

To verify a preview, open the three page URLs directly. In Explore, compare *cat* for Humans and GPT-5 mini, switch connections, then change views. In Lexical decision, the GPT-5 mini Table 6 row should show 0.866 for the priming effect and −0.667 for activation versus RT. In Implicit bias, switch from the editable priming experiment to Association paths. [Vercel Flask deployment reference](https://vercel.com/docs/frameworks/backend/flask)

## Data and refresh procedure

| View | Backing store | Meaning |
|---|---|---|
| Explorer | `resources/explore/` under `wordviz/` | Counts pooled over the three cleaned response slots. Forward follows cue to response. Backward follows response to source cue. Merged assigns each neighbor the maximum reciprocal count. Percentages use the sum of weights in the chosen direction. |
| Lexical decision and gender priming | `resources/experiments.sqlite` | Fully normalized saved-matrix values for the 50 LDT triplets and all available gender-target rows. The UI performs signed-rank tests on the displayed experiment. Table 6 includes the Spearman correlation with measured RT. |
| Association paths | `resources/paths/` | Unweighted shortest-hop paths on each filtered undirected network, with NRC VAD v2.1 word valence. This uses the version of the graph held in the repository, which can differ from the graph used for the archived activation matrices. |

Color in the explorer identifies the word/model selection. Path colors encode valence. The bias views cover Figures 3–9. The identity and political-party experiments in Figures 10–12 are deferred.

To refresh after upstream research data change, run these from the repository root using the **research** environment with pandas, NumPy and SciPy installed and Git LFS data materialized:

```sh
python scripts/build_explorer_store.py
python scripts/build_experiment_store.py
python scripts/build_path_store.py
```

Run the regression suite from the repository root using a Python with the web dependencies:

```sh
python -m unittest discover -s tests -p 'test_website*.py'
```
