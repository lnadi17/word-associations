"""Lexical decision, priming contrasts and target activation heatmaps."""

from __future__ import annotations

from functools import lru_cache
from math import hypot
from html import escape
from textwrap import wrap

from dash import Dash, Input, Output, State, dcc, html, set_props
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .experiment_data import (
    DEFAULT_PAIRS, EXACT_SIGN_RANK_LIMIT, MODELS, contrast_signed_rank, ldt_stats, metadata, signed_rank, two_sided_signed_rank,
)
from .social_data import domain_info, domain_suggest, domain_values
from .social_domains import DOMAIN_PRIMES, key_for, label_for, primes_for, resolve_domain
from .paired_presets import paired_presets
from .target_presets import TARGET_PRESETS
from .model_options import model_options
from .model_inclusion import included_models

ACCENT = "#17735f"
SECONDARY = "#bf6b36"
GRID = "#e4e8e5"
INK = "#20342d"
GROUPS = ["Female-related targets", "Male-related targets"]
PRIME_TEXT = ", ".join(f"{left}/{right}" for left, right in DEFAULT_PAIRS)
BIAS_PAPER = "https://arxiv.org/html/2510.24488#S3"
GRAPH_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


def field(label, control, persistence=True):
    # Dash unmounts the controls when the user switches experiment views.
    control.persistence = persistence
    control.persistence_type = "memory"
    return html.Div([html.Label(label, htmlFor=control.id, id=control.id + "-label"), control], className="field")


def model_fields(prefix):
    options = model_options(MODELS)
    return [
        field("Network", dcc.Dropdown(id=prefix + "-a", options=options, optionHeight=56, value="Humans", clearable=False)),
        field("Compare with", dcc.Dropdown(id=prefix + "-b", options=options, optionHeight=56, placeholder="Choose another network")),
    ]


def selected_models(a, b):
    return list(dict.fromkeys(model for model in (a, b) if model in MODELS))


def style_figure(fig, height):
    fig.update_layout(
        template="plotly_white", height=height, font={"family": "Inter, Arial, sans-serif", "color": INK},
        paper_bgcolor="rgba(0,0,0,0)", margin={"t": 38, "b": 60, "l": 70, "r": 25},
        legend={"orientation": "h", "y": 1.10}, hoverlabel={"namelength": -1},
    )
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def model_axis(figure, models):
    labels = [MODELS[m] for m in models]
    lines = [wrap(label, width=14, break_long_words=False) for label in labels]
    ticks = ["<br>".join(escape(part) for part in parts) for parts in lines]
    figure.update_xaxes(type="category", tickmode="array", tickvals=labels, ticktext=ticks, tickangle=0,
                        tickfont={"size": 11}, automargin=False)
    figure.update_layout(margin_b=32 + 16 * max(map(len, lines)), margin_r=12)


def graph(fig):
    return dcc.Graph(figure=fig, responsive=True, config=GRAPH_CONFIG,
                     style={"height": f"{fig.layout.height}px", "minWidth": 0})


def result_table(headers, rows, selected=(), label="Results"):
    return html.Div(html.Table([
        html.Thead(html.Tr([html.Th(text, scope="col") for text in headers])),
        html.Tbody([html.Tr(
            [html.Th(html.Strong(row[0]) if row[0] in selected else row[0], scope="row"),
             *[html.Td(value) for value in row[1:]]],
            className="selected-row" if row[0] in selected else "",
        ) for row in rows]),
    ], className="data-table"), className="table-wrap", style={"overflowX": "auto"},
        tabIndex=0, role="region", **{"aria-label": label})


def format_p(p):
    if p is None:
        return "—"
    if p == 0:
        return "< 1e−16"
    return f"{p:.2e}" if p < .001 else f"{p:.3f}"


def provenance(models, task):
    sources = metadata("provenance")
    return html.Details([
        html.Summary("Matrix sources"),
        html.Ul([html.Li(
            f"{sources[model][task]['file']} · {sources[model][task]['nodes']:,} nodes · "
            f"SHA-256 {sources[model][task]['sha256'][:12]}"
        ) for model in models]),
    ])


def ldt_layout():
    triples = metadata("ldt")["triplets"]
    return html.Div([
        html.Div([
            *model_fields("ldt"),
            field("Highlight triplet", dcc.Dropdown(id="ldt-triplet", clearable=True,
                options=[{"value": i, "label": f"{t} · {r} / {u}"} for i, (t,r,u,*_) in enumerate(triples)],
                placeholder="Choose a target and primes")),
            html.Button("Run tests", id="ldt-run", n_clicks=0, className="primary"),
        ], className="controls"),
        dcc.Loading(html.Div(id="ldt-results")),
    ])


def render_ldt(a, b, target_index):
    if a == b and a in MODELS:
        return html.P("Choose a different comparison network, or clear Compare with.", role="status")
    models = selected_models(a, b)
    if not models:
        return html.P("Choose a network.")
    experiment = metadata("ldt")
    triples = experiment["triplets"]
    figure = go.Figure()
    for i, (condition, color) in enumerate([("Related", ACCENT), ("Unrelated", SECONDARY)]):
        xs, ys, detail = [], [], []
        for model in models:
            for trip, values in zip(triples, experiment["activation"][model]):
                if values is None:
                    continue
                xs.append(MODELS[model])
                ys.append(values[i])
                detail.append([trip[0], trip[i+1], trip[i+3]])
        figure.add_trace(go.Box(
            x=xs, y=ys, name=condition, boxpoints="all", jitter=.2, pointpos=0,
            marker={"color": color, "size": 4}, line={"color": color},
            customdata=detail,
            hovertemplate="%{customdata[1]} → %{customdata[0]}<br>Activation %{y:.4f}<br>RT %{customdata[2]:.3f}<extra>%{fullData.name}</extra>",
            offsetgroup=condition,
        ))
    selected = isinstance(target_index, int) and 0 <= target_index < len(triples)
    trip_detail = []
    if selected:
        trip = triples[target_index]
        observed = {m: experiment["activation"][m][target_index] for m in models
                    if experiment["activation"][m][target_index] is not None}
        for i, condition in enumerate(["Related", "Unrelated"]):
            figure.add_trace(go.Box(
                x=[MODELS[m] for m in observed], y=[pair[i] for pair in observed.values()],
                name=f"{trip[i+1]} → {trip[0]}", offsetgroup=condition, boxpoints="all",
                pointpos=0, line={"width": 0}, marker={"color": INK, "symbol": "diamond-open", "size": 13},
                showlegend=False, hovertemplate="%{y:.4f}<extra>%{fullData.name}</extra>"))
        trip_rows = []
        for i, condition in enumerate(("Related", "Unrelated")):
            trip_rows.append([
                condition, trip[i + 1], f"{trip[i + 3]:.3f}",
                *[f"{observed[model][i]:.4f}" if model in observed else "Unavailable" for model in models],
            ])
        trip_detail = [
            html.H3(f"Selected target · {trip[0]}"),
            result_table(["Condition", "Prime", "Human RT (standardized)",
                          *[f"{MODELS[m]} activation" for m in models]],
                         trip_rows, label=f"Observed RT and activation for {trip[0]}"),
        ]
    style_figure(figure, 390).update_layout(boxmode="group", yaxis_title="Normalized target activation")
    model_axis(figure, models)
    table = []
    for model in MODELS:
        effect, pw, rho, ps = ldt_stats(model, experiment)
        count = sum(pair is not None for pair in experiment["activation"][model])
        table.append([MODELS[model], count, f"{effect:.3f}" if effect is not None else "—", format_p(pw),
                      2*count, f"{rho:.3f}" if rho is not None else "—", format_p(ps)])
    return [
        html.Section([html.H2("Target activation"), html.P(
            "LWOW Figure 3 · Each point is the normalized target activation for one prime–target condition."
        ), graph(figure), *trip_detail], className="panel"),
        html.Section([html.H2("Table 6 · Priming and reaction time"),
            result_table(["Network", "Pairs", "Wilcoxon effect", "Wilcoxon p", "RT observations",
                          "Spearman ρ", "Correlation p"], table, [MODELS[m] for m in models],
                         label="Table 6 priming effects and correlations"),
            html.P("One-sided signed-rank test of related minus unrelated activation, with effect z/√N. "
                   "Spearman’s ρ compares both conditions with their measured human reaction times. Pairs count triplets with the target and both primes in the network."),
            provenance(MODELS, "LDT"),
            html.A("Published Figure 3 and Table 6", href="https://www.nature.com/articles/s41597-025-05156-9#Fig3"),
        ], className="panel"),
    ]


def initial_primes(domain):
    if domain == "gender":
        return [left for left, _ in DEFAULT_PAIRS] + [right for _, right in DEFAULT_PAIRS]
    return list(primes_for(domain))


def target_field(label, words, index, persistence):
    return field(label, dcc.Dropdown(id=f"bias-targets-{index}", multi=True,
                                     options=list(words), value=list(words),
                                     placeholder="Search target words"), persistence=persistence)


def preset_words(domain, name):
    if not isinstance(name, str):
        return ()
    if isinstance(domain, dict):
        return domain["lists"].get(name, ())
    preset = TARGET_PRESETS.get(domain, {}).get(name)
    return preset.words if preset else ()


def named_choice(domain, value):
    """Prefix option values so arbitrary saved list names cannot collide with a design."""
    if isinstance(value, str) and value.startswith("pair/"):
        return paired_presets(domain).get(value[5:]), None
    return None, value.removeprefix("list/") if isinstance(value, str) else None


def target_set_label(pair, side, words=None):
    label = pair["labels"][side]
    if pair.get("manual_targets"):
        return ("First target list", "Second target list")[side]
    if not pair.get("survey_nominations"):
        return f"Target words in “{label}”"
    if words is None:
        return f"Target words for {label}"
    if set(words) == set(pair["groups"][side]):
        return f"Target words associated with {label}"
    return f"Selected target words for {label}"


def prime_controls(method, domain):
    primes = initial_primes(domain)
    scope = key_for(domain)
    if method == "heat":
        return [field("Primes", dcc.Dropdown(id="heat-primes", options=primes,
                      value=primes, multi=True), persistence=scope)]
    if domain == "gender":
        return [field("Prime pairs (first/second)", dcc.Textarea(id="effect-pairs",
                      value=PRIME_TEXT, rows=2, style={"width": "100%"}), persistence=scope)]
    return [
        field(label, dcc.Dropdown(id=f"effect-{side}", options=primes,
              value=primes[i] if isinstance(domain, dict) else None, placeholder="Choose a prime"),
              persistence=scope)
        for i, (side, label) in enumerate((("left", "First prime"), ("right", "Second prime")))
    ]


def study_target_editor(domain, name, method, session_targets):
    domain = resolve_domain(domain)
    if not domain or not isinstance(name, str):
        return html.P("Choose a target-word preset.", role="status")
    pair, list_name = named_choice(domain, name)
    persistence = f"{key_for(domain)}/{name}"
    if pair:
        primes, groups = pair["primes"], pair["groups"]
        saved = session_targets.get(persistence) if isinstance(session_targets, dict) else None
        if isinstance(saved, list) and len(saved) == 2 and all(
                isinstance(group, list) and all(isinstance(w, str) for w in group) for group in saved):
            groups = saved
        note = [pair["context"]]
        if pair.get("url"):
            note.extend([" ", html.A(pair["citation"], href=pair["url"])])
        if pair.get("data_url") and pair["data_url"] != pair.get("url"):
            note.extend([" · ", html.A("Source data", href=pair["data_url"])])
        if pair.get("evidence_url"):
            note.extend([" · ", html.A("Target rankings and nomination counts (CSV)",
                                       href=pair["evidence_url"], download="")])
        controls = [
            dcc.Store(id="paired-targets", data=name),
            dcc.Store(id="temporary-target-key", data={"pair": persistence}),
            html.Div([target_field(f"{target_set_label(pair, i)} (expected prime “{prime}”)", words, i, False)
                      for i, (prime, words) in enumerate(zip(primes, groups))],
                     className="controls"),
        ]
    else:
        words = preset_words(domain, list_name)
        if not words:
            return html.P("Choose a target-word preset.", role="status")
        if isinstance(domain, dict):
            note = f"Single-list exploratory contrast from {domain['name']}."
        else:
            study = TARGET_PRESETS[domain][list_name].study
            note = ["Single-list exploratory contrast. ", study.context, " ", html.A(study.citation, href=study.url)]
        controls = [html.Div([target_field("Target words", words, 0, persistence),
                              *prime_controls(method, domain)], className="controls")]
    return html.Div([
        *controls, html.P(note, className="source-note"),
        html.P(id="target-edits", className="source-note", role="status"),
        html.Button("Restore preset words", id="target-reset", n_clicks=0),
    ], key=persistence)


def stimulus_editor(method, domain, session_targets=None):
    persistence = key_for(domain)
    if domain == "gender":
        fields = [target_field(label, words, i, domain)
                  for i, (label, words) in enumerate(zip(GROUPS, metadata("targets")))]
        fields.extend(prime_controls(method, domain))
        fields.append(html.Button("Restore paper preset", id=f"{method}-reset", n_clicks=0))
        note = ("Enter ordered prime pairs as first/second, separated by commas."
                if method == "effect" else "Choose which saved prime columns to display.")
        return html.Details([html.Summary("Edit primes and targets"), html.P(note),
                             html.Div(fields, className="controls")], className="panel")
    names = list(domain["lists"] if isinstance(domain, dict) else TARGET_PRESETS[domain])
    designs = paired_presets(domain)
    pairing_hint = []
    if isinstance(domain, dict) and not designs:
        pairing_hint = [html.P("For a two-set comparison across all available networks, edit this domain "
                               "and assign a separate target list to each of two primes.", className="hint")]
    if not names and not designs:
        words = session_targets.get(persistence) if isinstance(session_targets, dict) else None
        if not isinstance(words, list) or not all(isinstance(word, str) for word in words):
            words = []
        return html.Section([
            dcc.Store(id="temporary-target-key", data=persistence),
            html.H3("Primes and targets"), *pairing_hint,
            html.Div([target_field("Target words", words, 0, False), *prime_controls(method, domain)],
                     className="controls study-primes"),
            html.P("Priming effects and Heatmaps share these target words for this visit.",
                   className="source-note"),
        ], className="panel")
    options = [{"label": name, "value": "pair/" + name} for name in designs]
    options.extend({"label": name, "value": "list/" + name}
                   for name in names)
    return html.Section([
        dcc.Store(id="target-method", data=method),
        html.H3("Primes and targets"), *pairing_hint,
        field("Target sets", dcc.Dropdown(id="target-preset", options=options,
              value=options[0]["value"], clearable=False, searchable=False), persistence=persistence),
        html.Div(id="target-editor"),
    ], className="panel")


def activation_layout(method, domain, session_targets):
    return html.Div([
        dcc.Store(id=f"{method}-domain", data=domain),
        dcc.Store(id="target-domain", data=domain),
        stimulus_editor(method, domain, session_targets),
        dcc.Loading(html.Div(id=f"{method}-results")),
    ], key=f"{method}-{key_for(domain)}")


def priming_layout(domain, session_targets=None):
    return activation_layout("effect", domain, session_targets)


def heatmaps_layout(domain, session_targets=None):
    return activation_layout("heat", domain, session_targets)


def parse_pairs(text, primes):
    if not text or len(text) > 512:
        raise ValueError("Enter at least one left/right prime pair.")
    pairs = []
    assigned = set()
    for entry in text.replace("\n", ",").split(","):
        sides = [part.strip().lower() for part in entry.split("/")]
        if len(sides) != 2 or any(part not in primes for part in sides):
            raise ValueError(f"Use two saved prime words separated by / for “{entry.strip()}”.")
        if sides[0] == sides[1] or assigned.intersection(sides):
            raise ValueError("Each prime must appear once in the paired test.")
        assigned.update(sides)
        pairs.append(tuple(sides))
    return pairs


def model_bars(group, cohort, models, effects, marks, detail):
    highlights = dict(zip(models, [ACCENT, SECONDARY]))
    labels = [MODELS[m] for m in cohort]
    ticks = ["<br>".join(escape(line) for line in wrap(label, width=23)) for label in labels]
    figure = go.Figure(go.Bar(
        y=labels, x=effects, orientation="h", name=group, text=marks, textposition="outside",
        cliponaxis=False, marker_color=[highlights.get(m, "#c4d1c9") for m in cohort], customdata=detail,
        hovertemplate="%{y}<br>Effect %{x:.3f}<br>Paired observations %{customdata[0]}<br>p %{customdata[1]}<extra></extra>",
    ))
    return style_figure(figure, max(330, 75 + 33 * len(labels))).update_layout(
        showlegend=False, margin={"t": 12, "b": 52, "l": 40, "r": 16},
        yaxis={"type": "category", "tickmode": "array", "tickvals": labels, "ticktext": ticks,
               "categoryorder": "array", "categoryarray": labels, "autorange": "reversed", "automargin": True},
        xaxis_title="Signed-rank effect (z/√N)",
    )


def effect_panels(headings, charts):
    has_negative = any(v is not None and v < 0 for fig in charts for v in fig.data[0].x)
    for fig in charts:
        fig.update_xaxes(range=[-1.1 if has_negative else 0, 1.1])
    return html.Div([
        html.Div([heading, graph(fig)], style={"minWidth": 0})
        for heading, fig in zip(headings, charts)
    ], style={"display": "grid", "gap": "16px",
              "gridTemplateColumns": "repeat(auto-fit, minmax(min(100%, 335px), 1fr))"})


def gender_effect(models, groups, values, pairs, canonical):
    # A model contributes when its matrix contains both primes of every selected pair.
    cohort = list(values)
    required = {prime for pair in pairs for prime in pair}
    available = metadata("provenance")
    eligible = [m for m in cohort if required.issubset(available[m]["gender"].get("primes", metadata("primes")))]
    rows, notes, charts = [], [], []
    if len(eligible) < len(cohort):
        notes.append(html.Li("Missing a selected prime: " + ", ".join(MODELS[m] for m in cohort if m not in eligible)))
    for group_index, (group, words) in enumerate(zip(GROUPS, groups)):
        shared = [word for word in words if eligible and all(word in values[m] for m in eligible)]
        missing = [word for word in words if word not in shared]
        notes.append(html.Li(f"{group}: {len(shared)}/{len(words)} targets. "
                             + ("Excluded: " + ", ".join(missing) + "." if missing else "All present.")))
        direction = (-1 if group_index else 1)
        effects, annotations, details = [], [], []
        for m in cohort:
            differences = [
                direction * (values[m][word][left] - values[m][word][right])
                for word in shared for left, right in pairs
            ] if m in eligible else []
            effect, p = signed_rank(differences)
            effects.append(effect)
            stars = "" if p is None else ("***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "")
            annotations.append(stars)
            details.append([len(differences), format_p(p)])
            rows.append([MODELS[m], group, len(differences), "—" if effect is None else f"{effect:.3f}", format_p(p)])
        charts.append(model_bars(group, cohort, models, effects, annotations, details))
    panels = effect_panels([html.H3(group) for group in GROUPS], charts)
    scope = ("Published gender word sets" if canonical else "Exploratory word sets") + (
        f" · common targets across {len(eligible)} eligible networks.")
    interpretation = (
        "Positive effects indicate stronger activation by female primes for female-related targets or male primes for male-related targets."
        if canonical else
        "Positive effects favor the first prime of each pair for female-related targets and the second prime for male-related targets."
    )
    return [html.P(scope + " " + interpretation + " The selected networks are highlighted."), panels,
            html.P("One-sided Wilcoxon test · * p < .05 · ** p < .01 · *** p < .001", className="source-note"),
            html.Ul(notes), html.Details([html.Summary("Test statistics"),
            result_table(["Network", "Target group", "Paired observations", "Effect", "One-sided p"], rows,
                         [MODELS[m] for m in models])])]


def heatmap(model, groups, primes, values, labels=None, missing_primes=()):
    labels = GROUPS if labels is None else labels
    title_lines = [wrap(label, width=26) for label in labels]
    titles = ["<br>".join(escape(line) for line in lines) for lines in title_lines]
    figure = make_subplots(rows=1, cols=len(groups), subplot_titles=titles, horizontal_spacing=.24)
    for col, words in enumerate(groups, 1):
        z = [[values[model].get(word, {}).get(prime) for prime in primes] for word in words]
        figure.add_trace(go.Heatmap(
            z=z, x=primes, y=words, coloraxis="coloraxis", hoverongaps=False,
            hovertemplate="%{x} → %{y}<br>Normalized activation %{z:.4f}<extra></extra>",
        ), row=1, col=col)
        figure.update_yaxes(autorange="reversed", type="category", showgrid=False, automargin=True,
                            tickmode="array", tickvals=words, row=1, col=col,
                            ticktext=[word if word in values[model] else word + " · N/A" for word in words])
        figure.update_xaxes(type="category", tickangle=-45, tickmode="array", tickvals=primes, automargin=True,
                           ticktext=[p + " · N/A" if p in missing_primes else p for p in primes],
                           row=1, col=col)
    top = 42 + 16 * max(map(len, title_lines))
    style_figure(figure, max(330, top + 95 + 18 * max(len(words) for words in groups)))
    figure.update_layout(
        margin={"t": top, "b": 85, "l": 115, "r": 65}, plot_bgcolor="#d6d8d6",
        coloraxis={"colorscale": [[0, "#f6f3e9"], [.5, "#67b7a0"], [1, "#123f39"]],
                   "cmin": 0, "cmax": 1, "colorbar": {"thickness": 10}},
    )
    return html.Div(
        html.Div(graph(figure), style={"minWidth": "740px" if len(groups) == 2 else "420px"}),
        style={"overflowX": "auto"}, tabIndex=0,
        **{"aria-label": f"{MODELS[model]} target heatmaps. Scroll horizontally on narrow screens."},
    )


def selection_problem(domain, a, b, groups, pair=None, highlights_only=False):
    if domain is None or (not highlights_only and (a not in MODELS or (b and b not in MODELS))):
        return "Choose a domain and a network."
    if not highlights_only and a == b:
        return "Choose a different comparison network, or clear Compare with."
    if pair and not all(groups):
        missing = [f"{label} (prime {prime})" for words, label, prime in zip(groups, pair["labels"], pair["primes"]) if not words]
        return "Choose target words for " + " and ".join(missing) + "."
    if not all(groups):
        return ("Choose at least one target word in each displayed group." if len(groups) > 1
                else "Search for a target word to run this analysis.")
    if any(len(group) > 100 for group in groups):
        return "Select up to 100 targets per group."
    if len(groups) == 2 and set(groups[0]).intersection(groups[1]):
        return "Assign each target to one group."
    if pair:
        self_targets = [p for p in pair["primes"] if any(p in group for group in groups)]
        if self_targets:
            return "Remove the following primes from the target words: " + ", ".join(self_targets) + "."
    return None


def selected_groups(domain, first, second, paired=False):
    return [list(dict.fromkeys(words or [])) for words in (
        (first, second) if domain == "gender" or paired else (first,)
    )]


def activation_sources(domain, models, infos=None):
    if domain == "gender":
        return provenance(models, "gender")
    notes = []
    for model in models:
        info = infos[model] if infos is not None else domain_info(domain, model)
        primes = ", ".join(info["primes"]) or "none"
        missing = ", ".join(info["missing_primes"])
        text = (f"{MODELS[model]} · {info['nodes']:,} graph nodes · {info['steps']} steps. "
                f"Available primes are {primes}.")
        if missing:
            text += f" The graph lacks {missing}."
        notes.append(html.Li([text, html.Br(), html.Small(
            f"{info['source']} · SHA-256 {info['sha256'][:12]}")]))
    return html.Details([html.Summary("Activation sources and coverage"),
                         html.P("These activations were calculated on the repository’s filtered association graphs."),
                         html.Ul(notes)])


def contrast_results(models, targets, left, right, values, highlights=()):
    shared = [w for w in targets if all(
        all(prime in values[m].get(w, {}) for prime in (left, right)) for m in models
    )]
    missing = [w for w in targets if w not in shared]
    coverage = f"{len(shared)}/{len(targets)} targets have both prime values in all {len(models)} eligible networks."
    if missing:
        coverage += " Excluded targets include " + ", ".join(missing) + "."
    if not shared:
        return [html.P(coverage, role="status")]
    rows, effects, details = [], [], []
    for model in models:
        deltas = [values[model][w][left] - values[model][w][right] for w in shared]
        effect, p, nonzero, method = two_sided_signed_rank(deltas)
        effects.append(effect)
        displayed_p = format_p(p) if p in (None, 0) else f"{p:.4g}"
        details.append([len(deltas), displayed_p])
        rows.append([MODELS[model], len(deltas), nonzero, f"{sum(deltas) / len(deltas):+.4f}",
                     "—" if effect is None else f"{effect:+.3f}", displayed_p, method])
    chart = []
    if any(effect is not None for effect in effects):
        figure = model_bars(f"{left} versus {right}", models, highlights, effects, [], details)
        figure.update_xaxes(range=[-1.1, 1.1])
        chart.append(graph(figure))
    return [
        html.P(f"Positive values indicate greater activation after {left}. Negative values favor {right}. "
               "The paired observations are the selected target words."),
        *chart,
        result_table(["Network", "Targets", "Nonzero differences", "Mean difference", "Effect", "Two-sided p", "P-value method"],
                     rows, selected=[MODELS[m] for m in highlights], label="Custom priming contrast"),
        html.P(coverage, className="source-note"),
        html.P(f"The exploratory Wilcoxon signed-rank test uses exact sign permutations for up to {EXACT_SIGN_RANK_LIMIT} "
               "nonzero differences and a tie-corrected normal approximation for larger samples. "
               "Zero differences drop out of the test. The effect uses all matched targets in z/√N. "
               "P-values are unadjusted. All-zero differences yield an undefined effect and p-value.",
               className="source-note"),
    ]


@lru_cache(maxsize=192)
def paired_network(model, primes, targets):
    # Keep coverage and target rows together to load each weighted network once.
    # Published non-gender stores already have each column normalized over all nodes.
    # Renormalizing each row over the chosen pair cancels the full-domain row factor.
    for domain, group in DOMAIN_PRIMES.items():
        if domain != "gender" and all(prime in group for prime in primes):
            info = domain_info(domain, model).copy()
            info.update(primes=[p for p in primes if p in info["primes"]],
                        missing_primes=[p for p in primes if p not in info["primes"]])
            rows = domain_values(domain, targets, [model])[model]
            values = {}
            for word, row in rows.items():
                pair = {p: row[p] for p in info["primes"]}
                length = hypot(*pair.values())
                values[word] = {p: v / length if length else 0.0 for p, v in pair.items()}
            return info, values
    domain = {"primes": primes}
    info = domain_info(domain, model)
    values = domain_values(domain, targets, [model])[model]
    return info, values


def paired_data(primes, groups, models):
    words = tuple(dict.fromkeys(w for group in groups for w in group))
    info, values = {}, {}
    for model in models:
        info[model], values[model] = paired_network(model, tuple(primes), words)
    return info, values


def holm_adjust(results):
    ordered = sorted((r for r in results if r["p"] is not None), key=lambda r: r["p"])
    previous = 0.0
    for i, result in enumerate(ordered):
        previous = max(previous, min(1.0, (len(ordered) - i) * result["p"]))
        result["adjusted"] = previous


def paired_effect(pair, groups, info, values, models):
    primes = pair["primes"]
    cohort = list(info)
    eligible = [m for m in cohort if not info[m]["missing_primes"]]
    labels = [target_set_label(pair, i, words) for i, words in enumerate(groups)]
    rows, notes, headings = [], [], []
    example = None
    for i, (label, words) in enumerate(zip(labels, groups)):
        preferred, other = primes[i], primes[1 - i]
        matched = [w for w in words if eligible and all(w in values[m] for m in eligible)]
        if matched and example is None:
            example = matched[0]
        headings.append(html.Header([
            html.H3(label),
            html.P(f"Positive bars mean stronger activation from “{preferred}”. "
                   f"Negative bars mean stronger activation from “{other}”."),
            html.Details([
                html.Summary(f"Show matched target words ({len(matched)})"),
                html.P(", ".join(matched)),
            ]) if matched else html.P("No matched target words.", role="status"),
        ], className="priming-panel-heading"))
        excluded = [w for w in words if w not in matched]
        coverage = f"{label} · {len(matched)}/{len(words)} targets shared across {len(eligible)} eligible networks."
        if excluded:
            coverage += " Excluded: " + ", ".join(excluded) + "."
        notes.append(html.Li(coverage))
        for model in cohort:
            deltas = [values[model][w][preferred] - values[model][w][other] for w in matched] if model in eligible else []
            effect, p, nonzero, method = contrast_signed_rank(deltas, directional=True)
            if model not in eligible:
                method = "Missing prime: " + ", ".join(info[model]["missing_primes"])
            elif not matched:
                method = "No common targets"
            rows.append(dict(model=model, group=label, preferred=preferred, count=len(deltas),
                             nonzero=nonzero, effect=effect, p=p, method=method, adjusted=None))
    holm_adjust(rows)
    charts = []
    for i, label in enumerate(labels):
        group_rows = rows[i * len(cohort):(i + 1) * len(cohort)]
        all_tied = [MODELS[r["model"]] for r in group_rows if r["count"] and not r["nonzero"]]
        if all_tied:
            notes.append(html.Li(f"{label} has only zero activation differences in " + ", ".join(all_tied)
                                 + ". The effect and p-value are undefined."))
        charts.append(model_bars(label, cohort, models, [r["effect"] for r in group_rows],
                      ["*" if r["adjusted"] is not None and r["adjusted"] < .05 else "" for r in group_rows],
                      [[r["count"], format_p(r["p"])] for r in group_rows]))
    table = [[MODELS[r["model"]], r["group"], r["preferred"], r["count"], r["nonzero"],
              "—" if r["effect"] is None else f'{r["effect"]:+.3f}', format_p(r["p"]),
              format_p(r["adjusted"]), r["method"]] for r in rows]
    missing = [f"{MODELS[m]} ({', '.join(info[m]['missing_primes'])})" for m in cohort if m not in eligible]
    intro = (f"The starting words (primes) are “{primes[0]}” and “{primes[1]}”. "
             "Both are tested on every matched target word. ")
    if pair.get("survey_nominations"):
        intro += "The original target lists group words that survey respondents associated more often with one group than with the other. "
    if example is not None:
        intro += f"For example, the network measures how much activation reaches “{example}” from each prime. "
    intro += "Each bar summarizes those differences across one target set in one network."
    return [
        html.P(intro), effect_panels(headings, charts),
        html.P("An asterisk marks a one-sided Wilcoxon test with Holm-adjusted p < .05. "
               "Each test compares the two primes within target words for one network.",
               className="source-note"),
        *([html.P("Networks missing a prime: " + ", ".join(missing) + ".", role="status")] if missing else []),
        html.Ul(notes),
        html.Details([html.Summary("Test statistics"), result_table(
            ["Network", "Target set", "Prime favored by a positive bar", "Matched targets", "Nonzero differences",
             "Effect", "One-sided p", "Holm p", "P-value method"], table, [MODELS[m] for m in models],
            label="Paired priming tests across networks"),
            html.P(f"Each eligible network uses the same matched words within a set. The test omits zero "
                   f"differences, averages tied ranks, and uses exact sign "
                   f"permutations through {EXACT_SIGN_RANK_LIMIT} nonzero differences. Larger samples use the "
                   "tie-corrected normal approximation. The effect divides z by the square root of all matched "
                   "targets. Holm correction covers the estimable tests in both sets across networks.")]),
        html.P("These exploratory tests describe associations in the selected networks. "
               "Activation is L2-normalized over graph nodes for each prime and then across these two primes.",
               className="source-note"),
    ]


def render_effect(domain, a, b, first, second, pair_text, left, right, paired_name=None, included=None):
    domain = resolve_domain(domain)
    cohort = included_models(included)
    if not cohort:
        return html.P("Select at least one network in Networks shown.", className="panel", role="status")
    pair = named_choice(domain, paired_name)[0] if domain and paired_name else None
    groups = selected_groups(domain, first, second, bool(pair))
    problem = selection_problem(domain, a, b, groups, pair, highlights_only=True)
    if problem:
        return html.P(problem, role="status")
    models = selected_models(a, b)
    if pair:
        info, values = paired_data(pair["primes"], groups, cohort)
        return html.Section([html.H2(f"{label_for(domain)} priming effects"),
                             *paired_effect(pair, groups, info, values, models),
                             activation_sources(domain, cohort, info)], className="panel")
    info = None
    if domain == "gender":
        try:
            pairs = parse_pairs(pair_text, DOMAIN_PRIMES[domain])
        except ValueError as err:
            return html.P(str(err), role="status")
        canonical = (all(set(chosen) == set(original) for chosen, original in zip(groups, metadata("targets")))
                     and set(pairs) == set(DEFAULT_PAIRS))
        values = domain_values(domain, groups[0] + groups[1], cohort)
        body = [*gender_effect(models, groups, values, pairs, canonical),
                html.A("Paper Figure 3 and methods", href=BIAS_PAPER)]
    else:
        if left not in primes_for(domain) or right not in primes_for(domain) or left == right:
            return html.P("Choose two different primes to compare their activation across the selected targets.",
                          className="panel", role="status")
        info = {m: domain_info(domain, m) for m in cohort}
        eligible = [m for m in cohort if left in info[m]["primes"] and right in info[m]["primes"]]
        body = [html.H3(f"{left} versus {right}")]
        if eligible:
            values = domain_values(domain, groups[0], eligible)
            body.extend(contrast_results(eligible, groups[0], left, right, values, models))
        absent = [f"{MODELS[m]} ({', '.join(p for p in (left, right) if p not in info[m]['primes'])})"
                  for m in cohort if m not in eligible]
        if absent:
            body.append(html.P("Networks missing a prime: " + ", ".join(absent) + ".", role="status"))
    return html.Section([html.H2(f"{label_for(domain)} priming effects"),
                         *body, activation_sources(domain, cohort, info)], className="panel")


def render_heatmaps(domain, a, b, first, second, selected_primes, paired_name=None):
    domain = resolve_domain(domain)
    pair = named_choice(domain, paired_name)[0] if domain and paired_name else None
    groups = selected_groups(domain, first, second, bool(pair))
    problem = selection_problem(domain, a, b, groups, pair)
    if problem:
        return html.P(problem, role="status")
    primes = list(pair["primes"]) if pair else list(dict.fromkeys(
        p for p in selected_primes or [] if p in primes_for(domain)))
    if not primes:
        return html.P("Select at least one prime for the heatmap.", className="panel", role="status")
    models = selected_models(a, b)
    info = None
    if pair:
        info, values = paired_data(primes, groups, models)
        labels = [target_set_label(pair, i, words) for i, words in enumerate(groups)]
    else:
        values = domain_values(domain, [w for words in groups for w in words], models)
        labels = GROUPS if domain == "gender" else ["Target words"]
    charts = []
    for model in models:
        missing = (info[model] if info else domain_info(domain, model))["missing_primes"]
        charts.append(html.H3(MODELS[model]))
        if pair and missing:
            charts.append(html.P("Paired heatmap unavailable. This graph lacks " + ", ".join(missing)
                                 + ".", role="status"))
        else:
            charts.append(heatmap(model, groups, primes, values, labels, missing))
    if domain == "gender":
        charts.append(html.A("Paper Figures 4–7 and methods", href=BIAS_PAPER))
    scope = "the two primes in this contrast" if pair else "the domain’s available primes"
    return html.Section([
        html.H2("Prime–target activation"),
        html.P(f"Each cell shows the target’s activation after a prime. Values run from 0 to 1 after "
               f"normalizing each prime over the graph and then each target over {scope}. "
               "Gray marks missing data."),
        *charts, activation_sources(domain, models, info),
    ], className="panel")


def register_stimuli(app):
    app.clientside_callback(
        """
        function(first, second, key, saved) {
            if (!key || !Array.isArray(first) || (key.pair && !Array.isArray(second))) {
                return dash_clientside.no_update;
            }
            const scope = key.pair || key;
            const words = key.pair ? [first, second] : first;
            if (JSON.stringify(saved?.[scope]) === JSON.stringify(words)) {
                return dash_clientside.no_update;
            }
            return {...saved, [scope]: words};
        }
        """,
        Output("session-targets", "data"),
        Input("bias-targets-0", "value", allow_optional=True),
        Input("bias-targets-1", "value", allow_optional=True),
        State("temporary-target-key", "data", allow_optional=True),
        State("session-targets", "data"),
    )

    for i in (0, 1):
        target = f"bias-targets-{i}"

        @app.callback(Output(target, "options"), Input(target, "search_value"), Input("social-a", "value"),
                      Input("social-b", "value"), Input(target, "value"), Input("included-models", "data"),
                      State("target-domain", "data"), State("bias-view", "value"))
        def suggestions(query, a, b, chosen, included, domain, method):
            models = included_models(included) if method == "priming" else selected_models(a, b)
            matches = domain_suggest(resolve_domain(domain), query, models)
            return list(dict.fromkeys([*(chosen or []), *matches]))

    app.callback(Output("target-editor", "children"), Input("target-domain", "data"),
                 Input("target-preset", "value"), Input("target-method", "data"),
                 State("session-targets", "data"))(study_target_editor)

    @app.callback(Output("target-edits", "children"), Input("bias-targets-0", "value"),
                  Input("bias-targets-1", "value", allow_optional=True),
                  State("target-domain", "data"), State("target-preset", "value"))
    def describe_target_edits(first, second, domain, name):
        domain = resolve_domain(domain)
        pair, single = named_choice(domain, name)
        original = pair["groups"] if pair else [preset_words(domain, single)]
        edited = [first, second] if pair else [first]
        if any(set(before) != set(after or []) for before, after in zip(original, edited)):
            return "Showing your edited target selection for this visit."
        return None

    @app.callback(Input("target-reset", "n_clicks"), State("target-domain", "data"),
                  State("target-preset", "value"), prevent_initial_call=True)
    def restore_study_words(clicks, domain, name):
        domain = resolve_domain(domain)
        pair, single = named_choice(domain, name)
        groups = pair["groups"] if pair else [preset_words(domain, single)]
        if clicks:
            for i, words in enumerate(groups):
                set_props(f"bias-targets-{i}", {"value": list(words)})

    def register_gender_reset(method):
        @app.callback(Input(f"{method}-reset", "n_clicks"), prevent_initial_call=True)
        def restore_gender(clicks):
            if clicks:
                for i, words in enumerate(metadata("targets")):
                    set_props(f"bias-targets-{i}", {"value": words})
                if method == "heat":
                    set_props("heat-primes", {"value": initial_primes("gender")})
                else:
                    set_props("effect-pairs", {"value": PRIME_TEXT})

    for method in ("effect", "heat"):
        register_gender_reset(method)


def register(app: Dash):
    @app.callback(Output("ldt-results", "children"), Input("ldt-a", "value"),
                  Input("ldt-b", "value"), Input("ldt-triplet", "value"), Input("ldt-run", "n_clicks"))
    def ldt_callback(a, b, target_index, _run):
        return render_ldt(a, b, target_index)

    register_stimuli(app)
    app.callback(Output("effect-results", "children"), Input("effect-domain", "data"),
                 Input("social-a", "value"), Input("social-b", "value"), Input("bias-targets-0", "value", allow_optional=True),
                 Input("bias-targets-1", "value", allow_optional=True),
                 Input("effect-pairs", "value", allow_optional=True),
                 Input("effect-left", "value", allow_optional=True),
                 Input("effect-right", "value", allow_optional=True),
                 Input("paired-targets", "data", allow_optional=True), Input("included-models", "data"))(render_effect)
    app.callback(Output("heat-results", "children"), Input("heat-domain", "data"),
                 Input("social-a", "value"), Input("social-b", "value"), Input("bias-targets-0", "value", allow_optional=True),
                 Input("bias-targets-1", "value", allow_optional=True), Input("heat-primes", "value", allow_optional=True),
                 Input("paired-targets", "data", allow_optional=True))(render_heatmaps)
