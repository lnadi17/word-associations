# Word Associations

A Python Dash application for exploring the project's cleaned trials, saved activation matrices, and filtered graphs. The analysis pages are available at `/`, `/lexical-decision`, and `/implicit-bias`. Open `/models` for collection identifiers, version evidence and dataset sources.

## Run locally

From the `website/` directory, use Python 3.13 and run

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

Open [the local explorer](http://127.0.0.1:8050/). Set `PORT` to choose a different port.

## Custom social domains

On **Implicit bias**, open **Social domain → Add a custom domain…**. Choose a research template to fill both prime words and both cited target sets, or enter your own name and 2–12 primes. To save a directional comparison, give each target list a name and an **Expected prime**. You can paste up to 100 targets per list, separated by commas or new lines. Choose **Save and use domain**. Use **Edit** to assign primes to existing lists or change the words. Definitions are stored in this browser.

Word valence uses all words in the graph with an NRC valence rating. Emotion word sets uses the eight NRC emotion categories. Both work with primes alone. For a custom domain, Priming effects and Heatmaps offer research sets when both exact prime words match a study. Otherwise, enter targets for each prime or assign saved lists. Association paths accepts individual target words or a saved list.

Priming effects, Word valence and Emotion word sets initially include all available networks. Open **Networks shown** to select whole families or individual builds. Those choices carry across the three views and are saved in this browser. **Highlight network** and **Second highlight** emphasize included networks in the plots and tables. In the valence and emotion matrices, each row is a network and each column is a prime or an emotion. Cells show signed estimates on a common color scale centered at zero. Hover or tap for sample sizes, p-values and, for valence, confidence intervals. Priming effects recomputes common target coverage for the included eligible networks. Each network’s valence and emotion estimates use its own covered vocabulary.

## Vercel deployment

Source-file deployments can supply a generated `release-files.json` with a public Git commit and SHA-256 hashes for research resources. The build restores those resources before Python imports the model registry. A full repository checkout already includes the website resources.

Use **website** as the Vercel project’s Root Directory for a Git deployment, or submit the contents of this directory as the deployment root. The Flask entry point is `app.py:app`. The build installs the web dependencies and runs `python prepare_static.py` to verify the research stores and refresh the pinned Plotly browser bundle in `public/vendor/`. Include the static assets and `wordviz/resources/` in the deployment source. Vercel serves the JavaScript and stylesheet from its CDN, while the function reads the research data. The six original Explore archives are bundled in the function. For each other network, Explore fetches the archive from the source commit on its first query in a new function instance. It checks the SHA-256 and byte count and atomically caches the verified file in its temporary directory. Graphs and activation statistics are bundled for every network. The Git checkout retains every archive for offline local use. The application uses no credentials, external database, or model API.

To verify a preview, open each page URL directly. In Explore, compare *cat* for Humans and GPT-5 mini, switch connections, then change views. In Lexical decision, the GPT-5 mini Table 6 row should show 0.866 for the priming effect and −0.667 for activation versus RT. In Implicit bias, switch from the editable priming experiment to Association paths. In Model details, expand an analyzed dataset and the partial GPT-5 nano collection to inspect their sources. [Vercel Flask deployment reference](https://vercel.com/docs/frameworks/backend/flask)

## Data and refresh procedure

| View | Backing store | Meaning |
|---|---|---|
| Explorer | `resources/explore/` under `wordviz/` | Counts pooled over the three cleaned response slots. Forward follows cue to response. Backward follows response to source cue. Merged assigns each neighbor the maximum reciprocal count. Percentages use the sum of weights in the chosen direction. |
| Lexical decision and gender priming | `resources/experiments.sqlite` | Fully normalized saved-matrix values for the 50 LDT triplets and all available gender-target rows. The UI performs signed-rank tests on the displayed experiment. Table 6 includes the Spearman correlation with measured RT. |
| Social priming effects and heatmaps | `resources/social/*.sqlite` | For religion, ethnicity, sexual orientation and political party, spread activation from each prime on the local weighted graph. Normalize the full domain matrix by column and then by row (L2). The user can edit targets and choose a contrast. |
| Custom domains | `resources/social_weights/` and `resources/paths/` | Propagate activation from any available graph node using the same weighted transition and step count as the published social models. Cache each computed prime column and normalize across the primes in the saved definition. |
| Word valence and emotion word sets | `resources/social_norms.json.gz` | Aggregate analyses across networks with packaged activations. Valence uses rated words from NRC VAD v1.0 with L2-normalized domain activations. Emotion uses eight NRC EmoLex v0.92 word sets with L1-normalized activations for the selected prime pair. Counts describe the covered vocabulary of each network. |
| Association paths | `resources/paths/` | Unweighted shortest-hop paths on each filtered undirected network, with NRC VAD v2.1 word valence. This uses the version of the graph held in the repository, which can differ from the graph used for the archived activation matrices. |

Priming effects and Heatmaps share the chosen design and target edits during a visit. The published gender analysis uses the archived two sets of 25 traits and five prime pairs. Each new comparison uses two distinct target sets and one ordered prime pair on the current weighted graphs. Both bars for a network summarize within-word differences. For the first set the difference is first-prime minus second-prime activation; for the second set the order reverses. Each set uses the same covered targets across eligible networks. All eligible networks use the expanded comparison cohort. The selected networks are highlighted. The statistics table reports matched words, nonzero differences, signed effect size, raw one-sided Wilcoxon p and Holm-adjusted p across estimable tests in that design. Heatmaps show the underlying activations. These comparisons extend the method to new survey-derived stimuli. The rankings describe reported group associations in specific samples.

| Paired comparison | Source of target words |
|---|---|
| Religion and ethnicity | Group-specific free responses from Nicolas, Bai & Fiske. The study labels, graph primes, sample counts and extraction rules accompany each comparison. [Primary study](https://doi.org/10.1037/pspa0000312) |
| Sexual orientation | Experiment 2 nominations for men and women analyzed separately. The source distinguishes target gender, while the graph nodes encode only orientation. [Primary study](https://doi.org/10.1177/19485506231183467) |
| US political party | Descriptors of Democratic and Republican supporters with the same direction of association in both the 2016 and 2021 surveys. These include traits, issues and party symbols. [Primary study](https://doi.org/10.1017/psrm.2023.30) |

Every survey-based design links to its source data and a downloadable CSV of target ranks, nomination counts and denominators. The CSV documents the original survey-derived words when a user edits a target set. The older single-list contrasts are also available. They use a two-sided test over one set of words and the selected networks. Their sources are listed below.

| Domain | Target-word presets and sources |
|---|---|
| Religion | Perceived religiousness and aggression come from a US study of Christian and Muslim stereotypes. Perceived warmth comes from a multi-group French study. [Erentzen et al., Study 3](https://doi.org/10.1177/13684302221138036) · [Ginguené et al., Study 1](https://doi.org/10.1017/S1755048326100601) |
| Ethnicity | The perceived competence scale accompanies five optional lists of traits associated with group labels in English-language books from 1800–1999. The historical labels identify the original corpus contrasts. Users choose graph primes separately. [Fiske et al., Table 7](https://doi.org/10.1037/0022-3514.82.6.878) · [Charlesworth et al., Table 1](https://doi.org/10.1073/pnas.2121798119) |
| Sexual orientation | Four lists sample words US participants reported hearing about straight people, gay men, lesbian women and bisexual people. The straight and bisexual presets combine responses about men and women. [McGorray and Petsko, Tables 5–6](https://doi.org/10.1177/19485506231183467) |
| US political party | Separate lists use frequently reported descriptions of Democratic and Republican supporters that occur in the six original analysis networks. The study’s “open minded” uses the graph spelling “open-minded”. [Rothschild et al., Table 1](https://doi.org/10.1007/s11109-018-9457-5) |

Paired targets and their source notes are in `wordviz/research_identity_targets.py` and `wordviz/research_other_targets.py`. Each module names its offline extraction script. The website ships the derived sets. Single-list presets are in `wordviz/target_presets.py`. Word valence implements the approach of Figures 10–11 for religion, ethnicity and sexual orientation. The political-party emotion contrast follows Figure 12. The same methods on other domains are labeled exploratory. Path colors use the local signed NRC VAD v2.1 ratings. [Study methods](https://arxiv.org/html/2510.24488#S2)

The weighted graph snapshots supply the four non-gender domains. The six original datasets retain their archived LDT and gender matrices. Newly processed networks use matrices calculated from their filtered graphs. Table 6 reports the number of triplets with an available target and both primes in each network. The build records the source for each model. Table 2 prints *athiest*; the graph node used here is *atheist*.

The Model details catalog supplies dataset keys and labels. Explore and Association paths list a dataset once its Explore counts and path graph are packaged. The experiment views also require its entries in the experiment database, social activation database and matching graph weights. Catalogued partial collections remain available on Model details.

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
