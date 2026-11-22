# Word association explorer implementation plan

The initial release plan below records the original three-page scope. The subsequent social-domain release added Figures 10–12 methods and split the implicit associations page into five views. The [website guide](website/README.md) documents the shipped analyses, data sources and refresh commands.

Build a Python website with three pages: **Explore associations**, **Lexical decision**, and **Implicit bias**. Give each page a primary model selector and an optional comparison model. Preserve the selected words or stimuli when switching models. Ten datasets support directional association browsing and graph queries. Six have saved spreading-activation results. The model selector should show which views are available for each dataset.

## 1. Choose the data source for each question

| Question | Source | Coverage |
|---|---|---|
| What did a cue elicit? Which cues elicited a word? | Cleaned `cue,R1,R2,R3` trials from `data/processed/*_processed.csv` and `lexicon/LWOW_datasets/processed_datasets/FA_*.csv` | Humans, Gemma, GPT-5 mini, Haiku, Llama3, Mistral, Kona2 12B, Llama3.1 8B, Llama3.2 3B, Qwen2.5 7B |
| Which paths connect two words in the filtered network? | `data/graphs/igraphs/FA_*_ig.graphml` or the corresponding filtered edge list | All ten |
| How much activation reached a target after priming? | `data/LDT_analyses/FA_matrices/*_LDT.csv` and `*_gender.csv`. Rows are receiving nodes. Columns are primes. | Humans, Gemma, GPT-5 mini, Haiku, Llama3, Mistral |
| Which effects should the canonical analysis reproduce? | `LDT_50_triplets.csv`, `gender_primes.csv`, `gender_targets.csv`, and `Statistical_tests/*.csv` under `data/LDT_analyses/` | Same six |

Use a small dataset manifest with stable IDs, display names, source paths, versions or fingerprints, and available analyses. Keep Llama3 distinct from Llama3.1 8B and Llama3.2 3B. Record graph and matrix provenance independently. For example, the saved Haiku gender matrix has 15,596 node rows and the current filtered graph has 15,066 nodes. Llama3 and Mistral also have different node sets across those artifacts. Label an activation plot with its matrix source and a path diagram with its graph source. A raw generation batch becomes eligible for a view when the required artifact exists. [Graph summary](data/summary_tables/FA_graphs_filtered_summary.csv) · [Saved matrices](data/LDT_analyses/FA_matrices/)

## 2. Explore associations

The top controls are **Model A**, optional **Model B**, a multi-word search field with removable words, **Forward / Backward / Merged**, and **Compact list / Bar list / 2D graph**. Start with one word. Adding another word compares words within the selected model. Adding Model B compares the same words across models. For multiple words and two models, align models in columns and give each query word its own row. Keep the direction, ranking measure, and display limit synchronized.

**Directional data.** Aggregate each nonblank cleaned response as one ordered pair `(cue, response)` across the three response positions. Forward on *cat* returns targets produced for *cat*. Backward returns source cues that elicited *cat*. Merged uses the greater of the two directional counts for each neighbor, with the total merged weight as its denominator. The filtered graph export combines reciprocal edges using their maximum count, so its endpoints cannot recover direction. For fast queries, materialize a disposable read-only lookup from the existing trials, indexed by both cue and response. This is a serving index over existing data. [Trial-to-edge logic](lwow/fa_networks.py)

**Measure and matching.** Show count plus percentage. For forward, divide by all nonblank response occurrences for that cue. For backward, divide by all occurrences of the selected response across source cues. Label those denominators in the interface. Thus 30% forward means 30% of responses to the cue, and 30% backward means 30% of occurrences of the word came from a given cue. Offer exact cleaned-word matching. Compute shared, A-only, and B-only membership over the full eligible neighborhoods before applying the display limit. With three or more selections, show a presence column per selection and filters for shared by all, some, or just one. Distinguish an observed zero from a missing cue, absent dataset, or undefined percentage.

| View | Single selection | Comparison |
|---|---|---|
| Compact list | Ranked word, count, share | Shared and exclusive sections with each model's count, share, rank, and percentage-point difference |
| Bar list | Sorted horizontal bars | Aligned paired bars on a common scale. Allow sorting by largest absolute difference as well as by either model’s share. |
| 2D graph | Focal words, neighboring words, directional edges, count/percentage on hover. Clicking a word makes it available as the next query. | Compute one layout on the union of displayed words and edges. Reuse neighbor positions in both panels. Distinguish each word–model selection by color and each focal word by shape. |

**What SWOW’s colors mean.** For *cat*, the live explorer’s color editor explicitly listed eight clusters, each with a representative word and member count. Examples were *feline* (amber, 20 words), *dog* (green, 10), *kitten* (red, 22), and *animal* (purple, 14). Frequency and in-degree were separate node-size settings. The UI confirms cluster membership as the color encoding. The particular clustering algorithm is unverified. The compact and bar comparisons also showed paired percentages and shared/exclusive sections for *cat* and *dog*. Our exact-match rule makes a shared word auditable across model vocabularies. [SWOW explorer](https://smallworldofwords.org/en/project/explore)

**Coloring in this release.** Assign colors by word–model selection. Draw directed arrows for forward and backward results and undirected links for merged results. Keep node placement consistent across comparisons.

## 3. Lexical decision task

Recreate **LWOW Figure 3** as two boxplots per network showing normalized activation at the same targets after a related or unrelated prime. Use the same vertical scale across model panels. A selected triplet should highlight its two observations and show its measured reaction times. For example, the target *britain* has related prime *england* and unrelated prime *like*. Put **Table 6 directly beneath the figure**, with one row per network and highlights for the selected models. [Published Figure 3](https://www.nature.com/articles/s41597-025-05156-9/figures/3) · [Published Table 6](https://www.nature.com/articles/s41597-025-05156-9/tables/6)

The **Run test** action uses the saved matrices. Reuse the numerical functions in `scripts/run_fa_statistical_tests.py`, extracting shared functions into `lwow/` during implementation where needed. L2-normalize each complete LDT matrix by columns and then rows before selecting any observations. Join the 50 triplets by target, related prime, and unrelated prime. Compute a one-sided paired Wilcoxon signed-rank test on related minus unrelated activation, with effect size *z*/√*N*. Compute Spearman’s ρ across the 100 matched activation and human reaction-time measurements. Display both p-values and sample sizes. Model comparisons use matched stimuli, axes, and condition order. These tests answer within-network questions. A difference between two models’ coefficients can be displayed descriptively. [Statistics implementation](scripts/run_fa_statistical_tests.py) · [Triplets and reaction times](data/LDT_analyses/LDT_50_triplets.csv)

**Table 6 acceptance values from the current repository** (rounded for display). Use full precision when testing. These are local results and can differ slightly from the publication.

| Network | Wilcoxon effect size | Spearman ρ, activation vs. RT |
|---|---:|---:|
| Humans | 0.870 | −0.627 |
| Gemma | 0.840 | −0.622 |
| GPT-5 mini | 0.866 | −0.667 |
| Haiku | 0.866 | −0.662 |
| Llama3 | 0.869 | −0.614 |
| Mistral | 0.859 | −0.615 |

[Saved effect sizes and p-values](data/LDT_analyses/Statistical_tests/wilcoxonLDT.csv) · [Saved correlations and p-values](data/LDT_analyses/Statistical_tests/corrLDT.csv)

## 4. Implicit bias figures and editable stimuli

| Paper figure | Website behavior | Available inputs |
|---|---|---|
| **3** | Two effect-size panels, one per target group. Show the one-sided paired Wilcoxon effect, p-value, and number of included pairs. Orient both panels as stereotype-consistent minus inconsistent. The repository stores female-prime minus male-prime for both groups, so reverse the sign of the male-target effect and retain its original one-sided p-value. | Six gender matrices and saved tests |
| **4, 5, 6, 7** | One heatmap template with presets for Humans, Mistral, Llama3, and Haiku respectively, also selectable for Gemma and GPT-5 mini. Show female-related and male-related target rows in separate panels, each with all ten prime columns. Comparing two networks locks row order, column order, and color scale. | Six gender matrices |
| **8, 9** | One path view with Humans and Haiku as the paper defaults. Display the four combinations of *feminine* and *masculine* with *compassionate* and *forceful*. Allow either endpoint to be replaced with a graph node. Display the union of minimum-hop paths and their length. Color positive valence blue and negative red. Use a separate symbol for missing ratings. | All ten filtered graphs plus local NRC VAD v2.1, whose signed scale is centered at zero |
| **10** | Defer the per-prime valence regression coefficients for religion, ethnicity, and sexual orientation. | The valence lexicon exists, but the saved matrices lack the complete identity-prime activation sets. |
| **11** | Defer the joint valence effect per identity domain, estimated from the multivariable model and its Wald statistic. | Requires the same missing sets of activation vectors. |
| **12** | Defer the eight emotion-specific Democrat/Republican contrasts. | Both political-prime activation columns and the paper’s eight discrete emotion word sets are missing. The available VAD lexicon measures different constructs. |

Figures 8–9 query the topology of the selected graph. Compute distances from each endpoint and keep edges on minimum-hop paths. This yields a compact union even when many routes share intermediate words. Label it as a reconstruction on the local graph with the local valence lexicon. A graph path can be shown for an endpoint that has never served as an activation prime. [Paper figures and methods](https://arxiv.org/html/2510.24488#S3) · [Mindset streams method](https://doi.org/10.1016/j.physa.2023.129074) · [Local valence lexicon](lexicon/NRC-VAD-Lexicon-v2.1.txt)

**Editing stimuli.** Provide a paper preset and an editable copy, with two target lists, ordered prime pairs, and reset-to-preset. Seed the five pairs as woman/man, female/male, mother/father, girl/boy, and feminine/masculine. Load the two lists of 25 target words directly from `gender_targets.csv`. A researcher can add, remove, or replace targets by searching matrix rows, and select primes from the saved columns for the active experiment. For custom heatmaps, allow unpaired prime selections. For a paired effect size, require explicit prime pairs and a defined contrast. Changes should recalculate the view from the selected saved values without running spreading activation. Keep LDT and gender matrices separate because their column sets define different row-normalization references. [Gender targets](data/LDT_analyses/gender_targets.csv) · [Prime list](data/LDT_analyses/gender_primes.csv)

**Coverage and comparisons.** The default gender effect uses the repository’s six-network common target set: 23 female-related and 25 male-related targets, giving 115 and 125 paired observations. Preserve all paper labels in the heatmaps, mark missing cells, and identify rows excluded from each test. For example, Gemma lacks *nagging* and *nurturing*. A custom model comparison must use the same eligible targets and prime pairs on both sides and report its own sample size. Normalize the full source matrix before filtering the displayed rows or columns. Label modified stimuli and cohorts as custom analyses.

## 5. Implementation order and release checks

1. **Data access and shared statistics.** Add the dataset manifest, validate schemas, build the directional lookup, and expose common functions for matrix normalization, keyed lookups, and tests. Cache the immutable source data by fingerprint. Verify defaults against the saved statistical CSVs before adding charts.
2. **Association explorer.** Build compact lists first, then paired bars, then the bounded 2D graph. Use Plotly for figures, Dash for Python callbacks and page state, and bounded graph queries over the saved networks. Only send the displayed subset to the browser. Validate navigation, keyboard-accessible lists, graph tooltips, and shared/exclusive membership under top-N changes. [Dash interaction model](https://dash.plotly.com/interactive-graphing) · [Plotly network figures](https://plotly.com/python/network-graphs/)
3. **LDT page.** Implement Figure 3 and Table 6 against the same analysis result. Verify all six rows, expected sample sizes, and the p-values in the existing outputs. Switching one comparison model must preserve the other model’s values.
4. **Bias page.** Build Figure 3, then the heatmaps and stimulus editor, then graph paths and valence coloring. Show an availability note for Figures 10–12. Graph-only models should open the path view directly and explain the missing activation matrices for other panels.
5. **Acceptance walkthrough.** Compare *cat* in Humans and GPT-5 mini. Humans has 54 observations of `cat → dog`. GPT-5 mini has 32 of `cat → purr` and 100 of `purr → cat`. Add *dog*, switch all three views, and verify that shared membership survives a display-limit change. Check the GPT-5 mini LDT row against Table 6. Swap a gender target, try a prime with no saved column, and confirm that no missing cell becomes zero. For the `feminine → forceful` path preset, the current Humans graph has 53 minimum-hop routes of length 3 and Haiku has 6 of length 4.
