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

## Custom social domains

On **Implicit bias**, open **Social domain → Add a custom domain…**. Name the domain, enter 2–12 prime words and add one or more named target lists (up to 100 words per list). Paste words separated by commas or new lines. You can start with primes from a published domain. Choose **Save and use domain** to analyze the definition or select it later from the same browser. Use **Edit** under the domain selector to update or delete it. Clearing this site's browser storage removes saved definitions.

Priming effects and Heatmaps use the selected target list. Association paths lets you choose a saved list and starts with up to two primes and two targets. Word valence uses all NRC-rated words in the network, and Emotion word sets uses the eight NRC categories for the selected prime pair. Graph coverage is reported for each network. Custom domains use the current weighted graph, including when they reuse a published gender prime.

## Vercel deployment

Use **website** as the Vercel project’s Root Directory for a Git deployment, or submit the contents of this directory as the deployment root. The Flask entry point is `app.py:app`. The build installs the web dependencies and runs `python prepare_static.py` to verify the research stores and refresh the pinned Plotly browser bundle in `public/vendor/`. Include the static assets and `wordviz/resources/` in the deployment source. Vercel serves the JavaScript and stylesheet from its CDN, while the function reads the research data. The application requires no credentials, external database, or model API.

To verify a preview, open the three page URLs directly. In Explore, compare *cat* for Humans and GPT-5 mini, switch connections, then change views. In Lexical decision, the GPT-5 mini Table 6 row should show 0.866 for the priming effect and −0.667 for activation versus RT. In Implicit bias, switch from the editable priming experiment to Association paths. [Vercel Flask deployment reference](https://vercel.com/docs/frameworks/backend/flask)

## Data and refresh procedure

| View | Backing store | Meaning |
|---|---|---|
| Explorer | `resources/explore/` under `wordviz/` | Counts pooled over the three cleaned response slots. Forward follows cue to response. Backward follows response to source cue. Merged assigns each neighbor the maximum reciprocal count. Percentages use the sum of weights in the chosen direction. |
| Lexical decision and gender priming | `resources/experiments.sqlite` | Fully normalized saved-matrix values for the 50 LDT triplets and all available gender-target rows. The UI performs signed-rank tests on the displayed experiment. Table 6 includes the Spearman correlation with measured RT. |
| Social priming effects and heatmaps | `resources/social/*.sqlite` | For religion, ethnicity, sexual orientation and political party, spread activation from each prime on the local weighted graph. Normalize the full domain matrix by column and then by row (L2). The user can edit targets and choose a contrast. |
| Custom domains | `resources/social_weights/` and `resources/paths/` | Propagate activation from any available graph node using the same weighted transition and step count as the published social models. Cache each computed prime column and normalize across the primes in the saved definition. |
| Word valence and emotion word sets | `resources/social_norms.json.gz` | Aggregate analyses across six networks. Valence uses rated words from NRC VAD v1.0 with L2-normalized domain activations. Emotion uses eight NRC EmoLex v0.92 word sets with L1-normalized activations for the selected prime pair. Counts describe the covered vocabulary of each network. |
| Association paths | `resources/paths/` | Unweighted shortest-hop paths on each filtered undirected network, with NRC VAD v2.1 word valence. This uses the version of the graph held in the repository, which can differ from the graph used for the archived activation matrices. |

Priming effects and Heatmaps share the selected preset and target edits for each domain during a visit. Choose a target set, then choose two primes to compare their activation across those words. Heatmaps show the observations for individual targets. The gender preset uses the original paper’s two trait lists. For the four other domains, the exploratory signed-rank test measures which prime activates the selected words more strongly. It reports an exact two-sided p-value for up to 20 nonzero paired differences and uses a tie-corrected normal approximation for larger samples.

| Domain | Target-word presets and sources |
|---|---|
| Religion | Perceived religiousness and aggression come from a US study of Christian and Muslim stereotypes. Perceived warmth comes from a multi-group French study. [Erentzen et al., Study 3](https://doi.org/10.1177/13684302221138036) · [Ginguené et al., Study 1](https://doi.org/10.1017/S1755048326100601) |
| Ethnicity | The perceived competence scale accompanies five optional lists of traits associated with group labels in English-language books from 1800–1999. The historical labels identify the original corpus contrasts. Users choose graph primes separately. [Fiske et al., Table 7](https://doi.org/10.1037/0022-3514.82.6.878) · [Charlesworth et al., Table 1](https://doi.org/10.1073/pnas.2121798119) |
| Sexual orientation | Four lists sample words US participants reported hearing about straight people, gay men, lesbian women and bisexual people. The straight and bisexual presets combine responses about men and women. [McGorray and Petsko, Tables 5–6](https://doi.org/10.1177/19485506231183467) |
| US political party | Separate lists use frequently reported descriptions of Democratic and Republican supporters that occur in all six networks. The study’s “open minded” uses the graph spelling “open-minded”. [Rothschild et al., Table 1](https://doi.org/10.1007/s11109-018-9457-5) |

Preset words, source citations and study contexts are defined in `wordviz/target_presets.py`. Word valence implements the approach of Figures 10–11 for religion, ethnicity and sexual orientation. The political-party emotion contrast follows Figure 12. The same methods on other domains are labeled exploratory. In these two charts the horizontal axis lists primes or emotion categories, and the vertical axis shows the valence slope or signed-rank effect. Path colors use the local signed NRC VAD v2.1 ratings. [Study methods](https://arxiv.org/html/2510.24488#S2)

The current graph snapshots supply the four new domains. Archived gender matrices supply the gender experiment. The build records the source for each model. Table 2 prints *athiest*; the graph node used here is *atheist*.

To refresh after upstream research data change, run these from the repository root using the **research** environment with pandas, NumPy and SciPy installed and Git LFS data materialized:

```sh
python scripts/build_explorer_store.py
python scripts/build_experiment_store.py
python scripts/build_path_store.py
```

To refresh the social analyses, use the research environment with python-igraph, SciPy, pandas and NumPy. The first command saves raw activation matrices for the aggregate norm analysis. The last command validates the path topology against the source graph and packages its edge weights. The custom norm views use verified copies of the same English NRC VAD v1.0 and EmoLex v0.92 inputs on the server. The build downloads these from the authors. For an offline build, run `website/prepare_static.py --vad /path/to/NRC-VAD-Lexicon.zip --emolex /path/to/NRC-Emotion-Lexicon-v0.92.zip` with the website Python environment. The lexicons stay out of Git and public assets. [NRC VAD](https://saifmohammad.com/WebPages/nrc-vad.html) · [NRC Emotion Lexicon](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm)

```sh
python scripts/build_social_store.py --raw-output /tmp/word-associations-social
python scripts/build_bias_norms.py --matrix-dir /tmp/word-associations-social \
  --vad /path/to/NRC-VAD-Lexicon.zip --emolex /path/to/NRC-Emotion-Lexicon-v0.92.zip
python scripts/build_custom_weights.py
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
