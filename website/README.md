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
| Social priming effects and heatmaps | `resources/social/*.sqlite` | For religion, ethnicity, sexual orientation and political party, spread activation from each prime on the local weighted graph. Normalize the full domain matrix by column and then by row (L2). The user can edit targets and choose a contrast. |
| Word valence and emotion word sets | `resources/social_norms.json.gz` | Aggregate analyses across six networks. Valence uses rated words from NRC VAD v1.0 with L2-normalized domain activations. Emotion uses eight NRC EmoLex v0.92 word sets with L1-normalized activations for the selected prime pair. Counts describe the covered vocabulary of each network. |
| Association paths | `resources/paths/` | Unweighted shortest-hop paths on each filtered undirected network, with NRC VAD v2.1 word valence. This uses the version of the graph held in the repository, which can differ from the graph used for the archived activation matrices. |

The gender priming preset retains the paper’s two trait lists. Other domains offer editable example words for exploring activation. A custom priming contrast answers which prime more strongly activates the selected targets. Its interpretation depends on how those targets were chosen. Word valence implements the approach of Figures 10–11 for religion, ethnicity and sexual orientation. The political-party emotion contrast follows Figure 12. The same methods on other domains are labeled exploratory. Path colors use the local signed NRC VAD v2.1 ratings. [Study methods](https://arxiv.org/html/2510.24488#S2)

The current graph snapshots supply the four new domains. Archived gender matrices supply the gender experiment. The build records the source for each model. Table 2 prints *athiest*; the graph node used here is *atheist*.

To refresh after upstream research data change, run these from the repository root using the **research** environment with pandas, NumPy and SciPy installed and Git LFS data materialized:

```sh
python scripts/build_explorer_store.py
python scripts/build_experiment_store.py
python scripts/build_path_store.py
```

To refresh the social analyses, use the research environment with python-igraph, SciPy, pandas and NumPy. The first command also saves raw activation matrices to a scratch directory for the second command. Supply the English NRC VAD v1.0 and EmoLex v0.92 source files from the authors. Only the aggregate norm results are included in the website. [NRC VAD](https://saifmohammad.com/WebPages/nrc-vad.html) · [NRC Emotion Lexicon](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm)

```sh
python scripts/build_social_store.py --raw-output /tmp/word-associations-social
python scripts/build_bias_norms.py --matrix-dir /tmp/word-associations-social \
  --vad /path/to/NRC-VAD-Lexicon.zip --emolex /path/to/NRC-Emotion-Lexicon.zip
```

Run the regression suite from the repository root using a Python with the web dependencies:

```sh
python -m unittest discover -s tests -p 'test_website*.py'
```

The browser regression checks measure the visible plotting area and exercise dropdowns at desktop and mobile widths. With the app running, use a test environment with Playwright installed:

```sh
python -m playwright install chromium webkit
python tests/browser_layout.py http://127.0.0.1:8050
python tests/browser_layout.py http://127.0.0.1:8050 --browser webkit
```
