"""Lexical decision, priming contrasts and target activation heatmaps."""

from __future__ import annotations

from math import erfc, sqrt

from dash import Dash, Input, Output, State, dcc, html, set_props
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .experiment_data import DEFAULT_PAIRS, MODELS, ldt_stats, metadata, signed_rank
from .social_data import domain_info, domain_suggest, domain_values
from .social_domains import DOMAIN_LABELS, DOMAIN_PRIMES

ACCENT = "#17735f"
SECONDARY = "#bf6b36"
GRID = "#e4e8e5"
INK = "#20342d"
GROUPS = ["Female-related targets", "Male-related targets"]
PRIME_TEXT = ", ".join(f"{left}/{right}" for left, right in DEFAULT_PAIRS)
EXAMPLE_TARGETS = ("book", "bread", "chair", "clock", "cup", "door", "key", "lamp", "pencil", "table")
BIAS_PAPER = "https://arxiv.org/html/2510.24488#S3"
GRAPH_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


def field(label, control, persistence=True):
    # Dash unmounts the controls when the user switches experiment views.
    control.persistence = persistence
    control.persistence_type = "memory"
    return html.Div([html.Label(label, htmlFor=control.id), control], className="field")


def model_fields(prefix):
    options = [{"value": m, "label": label} for m, label in MODELS.items()]
    return [
        field("Network", dcc.Dropdown(id=prefix + "-a", options=options, value="Humans", clearable=False)),
        field("Compare with", dcc.Dropdown(id=prefix + "-b", options=options, placeholder="Choose another network")),
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
        for i, condition in enumerate(["Related", "Unrelated"]):
            figure.add_trace(go.Box(
                x=[MODELS[m] for m in models], y=[experiment["activation"][m][target_index][i] for m in models],
                name=f"{trip[i+1]} → {trip[0]}", offsetgroup=condition, boxpoints="all",
                pointpos=0, line={"width": 0}, marker={"color": INK, "symbol": "diamond-open", "size": 13},
                showlegend=False, hovertemplate="%{y:.4f}<extra>%{fullData.name}</extra>"))
        trip_rows = []
        for i, condition in enumerate(("Related", "Unrelated")):
            trip_rows.append([
                condition, trip[i + 1], f"{trip[i + 3]:.3f}",
                *[f"{experiment['activation'][model][target_index][i]:.4f}" for model in models],
            ])
        trip_detail = [
            html.H3(f"Selected target · {trip[0]}"),
            result_table(["Condition", "Prime", "Human RT (standardized)",
                          *[f"{MODELS[m]} activation" for m in models]],
                         trip_rows, label=f"Observed RT and activation for {trip[0]}"),
        ]
    style_figure(figure, 390).update_layout(boxmode="group", yaxis_title="Normalized target activation")
    table = []
    for model in MODELS:
        effect, pw, rho, ps = ldt_stats(model, experiment)
        table.append([MODELS[model], len(triples), f"{effect:.3f}", format_p(pw),
                      2*len(triples), f"{rho:.3f}", format_p(ps)])
    return [
        html.Section([html.H2("Target activation"), html.P(
            "LWOW Figure 3 · Each point is the normalized target activation for one prime–target condition."
        ), graph(figure), *trip_detail], className="panel"),
        html.Section([html.H2("Table 6 · Priming and reaction time"),
            result_table(["Network", "Pairs", "Wilcoxon effect", "Wilcoxon p", "RT observations",
                          "Spearman ρ", "Correlation p"], table, [MODELS[m] for m in models],
                         label="Table 6 priming effects and correlations"),
            html.P("One-sided signed-rank test of related minus unrelated activation, with effect z/√N. "
                   "Spearman’s ρ compares both conditions with their measured human reaction times."),
            provenance(MODELS, "LDT"),
            html.A("Published Figure 3 and Table 6", href="https://www.nature.com/articles/s41597-025-05156-9#Fig3"),
        ], className="panel"),
    ]


def initial_groups(domain):
    return metadata("targets") if domain == "gender" else [list(EXAMPLE_TARGETS)]


def initial_primes(domain):
    if domain == "gender":
        return [left for left, _ in DEFAULT_PAIRS] + [right for _, right in DEFAULT_PAIRS]
    return list(DOMAIN_PRIMES[domain])


def stimulus_editor(method, domain):
    gender = domain == "gender"
    group_labels = GROUPS if gender else ["Target words"]
    fields = [
        field(label, dcc.Dropdown(id=f"{method}-targets-{i}", multi=True, value=words,
                                  placeholder="Search target words"), persistence=domain)
        for i, (label, words) in enumerate(zip(group_labels, initial_groups(domain)))
    ]
    primes = initial_primes(domain)
    if method == "heat":
        fields.append(field("Primes", dcc.Dropdown(id="heat-primes", options=primes,
                            value=primes, multi=True), persistence=domain))
    elif gender:
        fields.append(field("Prime pairs (first/second)", dcc.Textarea(id="effect-pairs",
                            value=PRIME_TEXT, rows=2, style={"width": "100%"}), persistence=domain))
    else:
        for key, label in (("left", "First prime"), ("right", "Second prime")):
            fields.append(field(label, dcc.Dropdown(id=f"effect-{key}", options=primes,
                                placeholder="Choose a prime"), persistence=domain))
    fields.append(html.Button("Restore paper preset" if gender else "Reset example",
                              id=f"{method}-reset", n_clicks=0))
    if gender:
        note = ("Enter ordered prime pairs as first/second, separated by commas."
                if method == "effect" else "Choose which saved prime columns to display.")
    else:
        note = ("The starting targets are ten everyday objects chosen to illustrate activation. "
                "Replace them with your study’s target words.")
    controls = [html.P(note), html.Div(fields, className="controls")]
    if gender:
        return html.Details([html.Summary("Edit primes and targets"), *controls], className="panel")
    return html.Section([html.H3("Primes and targets"), *controls], className="panel")


def activation_layout(method, domain):
    return html.Div([
        dcc.Store(id=f"{method}-domain", data=domain),
        stimulus_editor(method, domain),
        dcc.Loading(html.Div(id=f"{method}-results")),
    ], key=f"{method}-{domain}")


def priming_layout(domain):
    return activation_layout("effect", domain)


def heatmaps_layout(domain):
    return activation_layout("heat", domain)


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


def gender_effect(models, groups, values, pairs, canonical):
    # The published local tests use the intersection of all six networks.
    # A custom comparison keeps an identical sample across its selected models.
    cohort = list(MODELS) if canonical else models
    rows, notes, charts = [], [], []
    selected_colors = dict(zip(models, [ACCENT, SECONDARY]))
    for group_index, (group, words) in enumerate(zip(GROUPS, groups)):
        shared = [word for word in words if all(word in values[m] for m in cohort)]
        missing = [word for word in words if word not in shared]
        notes.append(html.Li(f"{group}: {len(shared)}/{len(words)} targets. "
                             + ("Excluded: " + ", ".join(missing) + "." if missing else "All present.")))
        direction = (-1 if group_index else 1)
        effects, annotations, details = [], [], []
        for m in cohort:
            differences = [
                direction * (values[m][word][left] - values[m][word][right])
                for word in shared for left, right in pairs
            ]
            effect, p = signed_rank(differences)
            effects.append(effect)
            stars = "" if p is None else ("***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "")
            annotations.append(stars)
            details.append([len(differences), format_p(p)])
            rows.append([MODELS[m], group, len(differences), "—" if effect is None else f"{effect:.3f}", format_p(p)])
        figure = go.Figure(go.Bar(
            x=[MODELS[m] for m in cohort], y=effects, name=group, text=annotations, textposition="outside",
            marker_color=[selected_colors.get(m, "#c4d1c9") for m in cohort], customdata=details,
            hovertemplate="%{x}<br>Effect %{y:.3f}<br>Paired observations %{customdata[0]}<br>p %{customdata[1]}<extra>%{fullData.name}</extra>",
        ))
        style_figure(figure, 330).update_layout(
            showlegend=False, margin={"t": 10, "b": 70, "l": 60, "r": 15},
            xaxis_tickangle=-25, yaxis_title="Signed-rank effect (z/√N)",
        )
        charts.append(figure)
    has_negative = any(v is not None and v < 0 for fig in charts for v in fig.data[0].y)
    for fig in charts:
        fig.update_yaxes(range=[-1.1 if has_negative else 0, 1.1])
    panels = html.Div([
        html.Div([html.H3(group), graph(fig)], style={"minWidth": 0})
        for group, fig in zip(GROUPS, charts)
    ], style={"display": "grid", "gap": "16px",
              "gridTemplateColumns": "repeat(auto-fit, minmax(min(100%, 335px), 1fr))"})
    scope = ("Paper preset · common targets across six networks." if canonical else
             "Custom analysis · common targets across the selected networks.")
    interpretation = (
        "Positive effects indicate stronger activation by female primes for female-related targets or male primes for male-related targets."
        if canonical else
        "Positive effects favor the first prime of each pair for female-related targets and the second prime for male-related targets."
    )
    return [html.P(scope + " " + interpretation), panels,
            html.P("One-sided Wilcoxon test · * p < .05 · ** p < .01 · *** p < .001", className="source-note"),
            html.Ul(notes), html.Details([html.Summary("Test statistics"),
            result_table(["Network", "Target group", "Paired observations", "Effect", "One-sided p"], rows,
                         [MODELS[m] for m in models])])]


def heatmap(model, groups, primes, values, labels=None, missing_primes=()):
    labels = GROUPS if labels is None else labels
    figure = make_subplots(rows=1, cols=len(groups), subplot_titles=labels, horizontal_spacing=.24)
    for col, words in enumerate(groups, 1):
        z = [[values[model].get(word, {}).get(prime) for prime in primes] for word in words]
        figure.add_trace(go.Heatmap(
            z=z, x=primes, y=words, coloraxis="coloraxis", hoverongaps=False,
            hovertemplate="%{x} → %{y}<br>Normalized activation %{z:.4f}<extra></extra>",
        ), row=1, col=col)
        figure.update_yaxes(autorange="reversed", type="category", showgrid=False,
                            tickmode="array", tickvals=words, row=1, col=col,
                            ticktext=[word if word in values[model] else word + " · N/A" for word in words])
        figure.update_xaxes(type="category", tickangle=-45, tickmode="array", tickvals=primes,
                           ticktext=[p + " · N/A" if p in missing_primes else p for p in primes],
                           row=1, col=col)
    style_figure(figure, max(330, 150 + 18 * max(len(words) for words in groups)))
    figure.update_layout(
        margin={"t": 55, "b": 85, "l": 115, "r": 65}, plot_bgcolor="#d6d8d6",
        coloraxis={"colorscale": [[0, "#f6f3e9"], [.5, "#67b7a0"], [1, "#123f39"]],
                   "cmin": 0, "cmax": 1, "colorbar": {"thickness": 10}},
    )
    return html.Div(
        html.Div(graph(figure), style={"minWidth": "740px" if len(groups) == 2 else "420px"}),
        style={"overflowX": "auto"}, tabIndex=0,
        **{"aria-label": f"{MODELS[model]} target heatmaps. Scroll horizontally on narrow screens."},
    )


def selection_problem(domain, a, b, groups):
    if domain not in DOMAIN_PRIMES or a not in MODELS or (b and b not in MODELS):
        return "Choose a domain and a network."
    if a == b:
        return "Choose a different comparison network, or clear Compare with."
    if not all(groups):
        return "Choose at least one target word in each displayed group."
    if any(len(group) > 100 for group in groups):
        return "Select up to 100 targets per group."
    if len(groups) == 2 and set(groups[0]).intersection(groups[1]):
        return "Assign each target to one group."
    return None


def selected_groups(domain, first, second):
    return [list(dict.fromkeys(words or [])) for words in (
        (first, second) if domain == "gender" else (first,)
    )]


def activation_sources(domain, models):
    if domain == "gender":
        return provenance(models, "gender")
    notes = []
    for model in models:
        info = domain_info(domain, model)
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


def contrast_results(models, targets, left, right, values):
    shared = [w for w in targets if all(
        all(prime in values[m].get(w, {}) for prime in (left, right)) for m in models
    )]
    missing = [w for w in targets if w not in shared]
    coverage = f"{len(shared)}/{len(targets)} targets have both prime values in every selected network."
    if missing:
        coverage += " Excluded targets include " + ", ".join(missing) + "."
    if not shared:
        return [html.P(coverage, role="status")]
    rows, effects = [], []
    for model in models:
        deltas = [values[model][w][left] - values[model][w][right] for w in shared]
        effect, _ = signed_rank(deltas)
        p = None if effect is None else erfc(abs(effect) * sqrt(len(deltas) / 2))
        effects.append(effect)
        rows.append([MODELS[model], len(deltas), sum(d != 0 for d in deltas),
                     f"{sum(deltas) / len(deltas):+.4f}", "—" if effect is None else f"{effect:+.3f}", format_p(p)])
    chart = []
    if any(effect is not None for effect in effects):
        figure = go.Figure(go.Bar(
            x=[MODELS[m] for m in models], y=effects, marker_color=[ACCENT, SECONDARY][:len(models)],
            hovertemplate="%{x}<br>Signed-rank effect %{y:+.3f}<extra></extra>",
        ))
        style_figure(figure, 330).update_layout(showlegend=False, yaxis_title="Signed-rank effect (z/√N)")
        figure.update_yaxes(range=[-1.1, 1.1])
        chart.append(graph(figure))
    return [
        html.P(f"Positive values indicate greater activation after {left}. Negative values favor {right}. "
               "The paired observations are the selected target words."),
        *chart,
        result_table(["Network", "Targets", "Nonzero differences", "Mean difference", "Effect", "Two-sided p"],
                     rows, label="Custom priming contrast"),
        html.P(coverage, className="source-note"),
        html.P("Exploratory two-sided Wilcoxon signed-rank test using the tie-corrected normal approximation. "
               "P-values are unadjusted. All-zero differences yield an undefined standardized effect and p-value.",
               className="source-note"),
    ]


def render_effect(domain, a, b, first, second, pair_text, left, right):
    groups = selected_groups(domain, first, second)
    problem = selection_problem(domain, a, b, groups)
    if problem:
        return html.P(problem, role="status")
    models = selected_models(a, b)
    if domain == "gender":
        try:
            pairs = parse_pairs(pair_text, DOMAIN_PRIMES[domain])
        except ValueError as err:
            return html.P(str(err), role="status")
        canonical = (all(set(chosen) == set(original) for chosen, original in zip(groups, metadata("targets")))
                     and set(pairs) == set(DEFAULT_PAIRS))
        values = domain_values(domain, groups[0] + groups[1], list(MODELS) if canonical else models)
        body = [*gender_effect(models, groups, values, pairs, canonical),
                html.A("Paper Figure 3 and methods", href=BIAS_PAPER)]
    else:
        if left not in DOMAIN_PRIMES[domain] or right not in DOMAIN_PRIMES[domain] or left == right:
            return html.P("Choose two different primes to compare their activation across the selected targets.",
                          className="panel", role="status")
        absent = []
        for model in models:
            available = domain_info(domain, model)["primes"]
            missing = [prime for prime in (left, right) if prime not in available]
            if missing:
                absent.append(f"{MODELS[model]} ({', '.join(missing)})")
        if absent:
            return html.P("This contrast requires prime columns missing from " + ", ".join(absent) + ".",
                          className="panel", role="status")
        values = domain_values(domain, groups[0], models)
        body = [html.H3(f"{left} versus {right}"), *contrast_results(models, groups[0], left, right, values)]
    return html.Section([html.H2(f"{DOMAIN_LABELS[domain]} priming effects"),
                         *body, activation_sources(domain, models)], className="panel")


def render_heatmaps(domain, a, b, first, second, selected_primes):
    groups = selected_groups(domain, first, second)
    problem = selection_problem(domain, a, b, groups)
    if problem:
        return html.P(problem, role="status")
    primes = list(dict.fromkeys(p for p in selected_primes or [] if p in DOMAIN_PRIMES[domain]))
    if not primes:
        return html.P("Select at least one prime for the heatmap.", className="panel", role="status")
    models = selected_models(a, b)
    values = domain_values(domain, [w for words in groups for w in words], models)
    charts = []
    for model in models:
        missing = [] if domain == "gender" else domain_info(domain, model)["missing_primes"]
        charts.extend([html.H3(MODELS[model]), heatmap(
            model, groups, primes, values, GROUPS if domain == "gender" else ["Target words"], missing)])
    if domain == "gender":
        charts.append(html.A("Paper Figures 4–7 and methods", href=BIAS_PAPER))
    return html.Section([
        html.H2("Prime–target activation"),
        html.P("Each cell shows the target’s activation after a prime. The scale runs from 0 to 1 "
               "after column and row normalization within each network and domain. Gray marks missing data."),
        *charts, activation_sources(domain, models),
    ], className="panel")


def register_stimuli(app, method):
    for i in (0, 1):
        target = f"{method}-targets-{i}"

        @app.callback(Output(target, "options"), Input(target, "search_value"), Input("social-a", "value"),
                      Input("social-b", "value"), Input(target, "value"), State(f"{method}-domain", "data"))
        def suggestions(query, a, b, chosen, domain):
            matches = domain_suggest(domain, query, selected_models(a, b))
            return list(dict.fromkeys([*(chosen or []), *matches]))

    @app.callback(Input(f"{method}-reset", "n_clicks"), State(f"{method}-domain", "data"),
                  prevent_initial_call=True)
    def reset(clicks, domain):
        if not clicks or domain not in DOMAIN_PRIMES:
            return
        for i, words in enumerate(initial_groups(domain)):
            set_props(f"{method}-targets-{i}", {"value": words})
        if method == "heat":
            set_props("heat-primes", {"value": initial_primes(domain)})
        elif domain == "gender":
            set_props("effect-pairs", {"value": PRIME_TEXT})
        else:
            set_props("effect-left", {"value": None})
            set_props("effect-right", {"value": None})


def register(app: Dash):
    @app.callback(Output("ldt-results", "children"), Input("ldt-a", "value"),
                  Input("ldt-b", "value"), Input("ldt-triplet", "value"), Input("ldt-run", "n_clicks"))
    def ldt_callback(a, b, target_index, _run):
        return render_ldt(a, b, target_index)

    for method in ("effect", "heat"):
        register_stimuli(app, method)
    app.callback(Output("effect-results", "children"), Input("effect-domain", "data"),
                 Input("social-a", "value"), Input("social-b", "value"), Input("effect-targets-0", "value"),
                 Input("effect-targets-1", "value", allow_optional=True),
                 Input("effect-pairs", "value", allow_optional=True),
                 Input("effect-left", "value", allow_optional=True),
                 Input("effect-right", "value", allow_optional=True))(render_effect)
    app.callback(Output("heat-results", "children"), Input("heat-domain", "data"),
                 Input("social-a", "value"), Input("social-b", "value"), Input("heat-targets-0", "value"),
                 Input("heat-targets-1", "value", allow_optional=True), Input("heat-primes", "value"))(render_heatmaps)
