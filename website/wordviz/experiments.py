"""LWOW lexical decision and editable gender-prime experiments."""

from __future__ import annotations

from dash import Dash, Input, Output, dcc, html
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .experiment_data import DEFAULT_PAIRS, MODELS, genders, ldt_stats, metadata, signed_rank, suggest_targets

ACCENT = "#17735f"
SECONDARY = "#bf6b36"
GRID = "#e4e8e5"
INK = "#20342d"
GROUPS = ["Female-related targets", "Male-related targets"]
PRIME_TEXT = ", ".join(f"{left}/{right}" for left, right in DEFAULT_PAIRS)
GRAPH_CONFIG = {"displaylogo": False, "responsive": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


def field(label, control):
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
    return dcc.Graph(figure=fig, config=GRAPH_CONFIG, style={"minWidth": 0})


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
    return f"{p:.2e}" if 0 < p < .001 else f"{p:.3f}"


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


def bias_layout():
    targets = metadata("targets")
    preset_primes = [p for pair in DEFAULT_PAIRS for p in pair]
    return html.Div([
        html.Div(model_fields("bias"), className="controls"),
        html.Details([
            html.Summary("Edit primes and targets"),
            html.Div([
                field("Female-related targets", dcc.Dropdown(id="bias-f-targets", options=targets[0],
                    value=targets[0], multi=True, placeholder="Search target words")),
                field("Male-related targets", dcc.Dropdown(id="bias-m-targets", options=targets[1],
                    value=targets[1], multi=True, placeholder="Search target words")),
                field("Heatmap primes", dcc.Dropdown(id="bias-primes", options=metadata("primes"),
                    value=preset_primes, multi=True)),
                field("Prime pairs for the effect test (left/right)", dcc.Textarea(id="bias-pairs",
                    value=PRIME_TEXT, rows=2, style={"width": "100%"})),
                html.Button("Restore paper preset", id="bias-reset", n_clicks=0),
            ], className="controls"),
            html.P("Use a comma between prime pairs. Targets can be any archived node. "
                   "Prime choices come from the ten saved activation columns."),
        ], className="panel"),
        dcc.Loading(html.Div(id="bias-results")),
    ])


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


def heatmap(model, groups, primes, values):
    figure = make_subplots(rows=1, cols=2, subplot_titles=GROUPS, horizontal_spacing=.24)
    for col, words in enumerate(groups, 1):
        z = [[values[model].get(word, {}).get(prime) for prime in primes] for word in words]
        figure.add_trace(go.Heatmap(
            z=z, x=primes, y=words, coloraxis="coloraxis", hoverongaps=False,
            hovertemplate="%{x} → %{y}<br>Normalized activation %{z:.4f}<extra></extra>",
        ), row=1, col=col)
        figure.update_yaxes(autorange="reversed", type="category", showgrid=False,
                            tickmode="array", tickvals=words, row=1, col=col,
                            ticktext=[word if word in values[model] else word + " · N/A" for word in words])
        figure.update_xaxes(type="category", tickangle=-45, row=1, col=col)
    style_figure(figure, max(330, 150 + 18 * max(len(words) for words in groups)))
    figure.update_layout(
        margin={"t": 55, "b": 85, "l": 115, "r": 65}, plot_bgcolor="#d6d8d6",
        coloraxis={"colorscale": [[0, "#f6f3e9"], [.5, "#67b7a0"], [1, "#123f39"]],
                   "cmin": 0, "cmax": 1, "colorbar": {"thickness": 10}},
    )
    return html.Div(
        html.Div(graph(figure), style={"minWidth": "740px"}),
        style={"overflowX": "auto"}, tabIndex=0,
        **{"aria-label": f"{MODELS[model]} target heatmaps. Scroll horizontally on narrow screens."},
    )


def render_bias(a, b, female, male, primes, pair_text):
    if a == b and a in MODELS:
        return html.P("Choose a different comparison network, or clear Compare with.", role="status")
    models = selected_models(a, b)
    groups = [list(dict.fromkeys(words or [])) for words in (female, male)]
    if not models or not all(groups):
        return html.P("Choose a network and at least one target in each group.", role="status")
    if any(len(group) > 100 for group in groups):
        return html.P("Select up to 100 targets per group.", role="status")
    if set(groups[0]).intersection(groups[1]):
        return html.P("Assign each target to one group.", role="status")
    values = genders(list(dict.fromkeys(groups[0] + groups[1])))
    known = metadata("primes")
    primes = list(dict.fromkeys(prime for prime in (primes or []) if prime in known))
    # Match the paper's female-prime and male-prime column blocks in the default heatmap.
    preset_order = [p for pair in DEFAULT_PAIRS for p in pair]
    if primes == preset_order:
        primes = [l for l, _ in DEFAULT_PAIRS] + [r for _, r in DEFAULT_PAIRS]
    try:
        pairs = parse_pairs(pair_text, known)
        # The estimand depends on group membership and directed pairs, not their display order.
        canonical = (all(set(chosen) == set(original) for chosen, original in zip(groups, metadata("targets")))
                     and set(pairs) == set(DEFAULT_PAIRS))
        effects = gender_effect(models, groups, values, pairs, canonical)
    except ValueError as err:
        effects = [html.P(str(err), role="status")]
    heatmaps = []
    if primes:
        for model in models:
            heatmaps.extend([html.H3(MODELS[model]), heatmap(model, groups, primes, values)])
    else:
        heatmaps.append(html.P("Select at least one saved heatmap prime."))
    return [
        html.Section([html.H2("Gender priming effects"), *effects], className="panel"),
        html.Section([html.H2("Prime–target activation"),
            html.P("Figures 4–7 · All panels use the same normalized scale from 0 to 1. Gray cells mark missing targets."),
            *heatmaps, provenance(models, "gender"),
            html.A("Published bias figures", href="https://arxiv.org/html/2510.24488#S3"),
        ], className="panel"),
    ]


def register(app: Dash):
    @app.callback(Output("ldt-results", "children"), Input("ldt-a", "value"),
                  Input("ldt-b", "value"), Input("ldt-triplet", "value"), Input("ldt-run", "n_clicks"))
    def ldt_callback(a, b, target_index, _run):
        return render_ldt(a, b, target_index)

    @app.callback(Output("bias-results", "children"), Input("bias-a", "value"), Input("bias-b", "value"),
                  Input("bias-f-targets", "value"), Input("bias-m-targets", "value"),
                  Input("bias-primes", "value"), Input("bias-pairs", "value"))
    def bias_callback(a, b, female, male, primes, pairs):
        return render_bias(a, b, female, male, primes, pairs)

    for side in ("f", "m"):
        target = f"bias-{side}-targets"

        @app.callback(Output(target, "options"), Input(target, "search_value"), Input("bias-a", "value"),
                      Input("bias-b", "value"), Input(target, "value"))
        def options(query, a, b, chosen):
            matches = suggest_targets(query, selected_models(a, b))
            return list(dict.fromkeys([*(chosen or []), *matches]))

    @app.callback(Output("bias-f-targets", "value"), Output("bias-m-targets", "value"),
                  Output("bias-primes", "value"), Output("bias-pairs", "value"),
                  Input("bias-reset", "n_clicks"), prevent_initial_call=True)
    def reset(_click):
        return (*metadata("targets"), [p for pair in DEFAULT_PAIRS for p in pair], PRIME_TEXT)
