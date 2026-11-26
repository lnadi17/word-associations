"""Valence regressions and emotion-word contrasts for social identity primes."""

from html import escape
from math import ceil
from textwrap import wrap

import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html

from .bias_norms import EMOTIONS, emotion_result, valence_result
from .experiments import ACCENT, SECONDARY, activation_sources, format_p, graph, result_table, style_figure
from .model_inclusion import included_models
from .models import MATRIX_MODELS
from .norm_lexicons import NormSourceError, load_lexicons
from .social_domains import key_for, label_for, primes_for, resolve_domain


def chart_selector(method):
    return html.Fieldset([
        html.Legend("Visualization", className="field-label"),
        dcc.RadioItems(id=f"{method}-chart-view", options=[
            {"label": "Table", "value": "matrix"}, {"label": "Scatter plot", "value": "scatter"},
        ], value="matrix", className="segmented", persistence=True, persistence_type="memory"),
    ], className="field norm-chart-selector")


def valence_layout(domain):
    return html.Div([dcc.Store(id="valence-domain", data=domain),
                     html.Div(chart_selector("valence"), className="controls"),
                     dcc.Loading(html.Div(id="valence-results"))])


def emotion_layout(domain):
    primes = primes_for(domain)
    fields = []
    for i, label in enumerate(("First prime", "Comparison prime")):
        control_id = ("emotion-left", "emotion-right")[i]
        fields.append(html.Div([
            html.Label(label, htmlFor=control_id),
            dcc.Dropdown(id=control_id, options=list(primes), value=primes[i], clearable=False,
                         persistence=key_for(domain), persistence_type="memory"),
        ], className="field"))
    return html.Div([dcc.Store(id="emotion-domain", data=domain),
                     html.Div([*fields, chart_selector("emotion")], className="controls"),
                     dcc.Loading(html.Div(id="emotion-results"))])


def number(value):
    return "—" if value is None else f"{value:.3f}"


CONTEXT = "#849491"


def tooltip_p(p):
    if p is None:
        return "p unavailable"
    for cutoff in (.001, .01, .05):
        if p < cutoff:
            return f"p < {cutoff}"
    return f"p = {p:.2f}"


def tooltip_estimate(label, value):
    if 0 < abs(value) < .005:
        if value > 0:
            return f"0 < {label.lower()} < 0.01"
        return f"−0.01 < {label.lower()} < 0"
    if value == 0:
        return f"{label} 0"
    estimate = f"{value:+.2f}".replace("-", "−")
    return f"{label} {estimate}"


def interval_endpoint(value):
    if value == 0:
        return "0"
    if abs(value) < .005:
        return "just above 0" if value > 0 else "just below 0"
    return f"{value:.2f}".replace("-", "−")


def norm_tooltip(model, row, value_key, prime_pair):
    if not row or row[value_key] is None:
        return None
    value = row[value_key]
    if value_key == "slope":
        category = row["prime"]
        heading = f"Prime · {category}"
        summary = "No linear trend in pleasantness."
        if value:
            direction = "higher" if value > 0 else "lower"
            summary = f"Stronger activation goes with {direction} pleasantness."
        metric = tooltip_estimate("Slope", value)
        if row["ci_low"] is not None and row["ci_high"] is not None:
            bounds = [interval_endpoint(row[key]) for key in ("ci_low", "ci_high")]
            metric += f" · 95% CI [{', '.join(bounds)}]"
        count = f"{row['n']:,} rated words"
    else:
        category = row["emotion"]
        first, second = prime_pair
        heading = f"{category.title()} · {first} vs {second}"
        summary = "Neither prime leads overall."
        if value:
            favored = first if value > 0 else second
            summary = f"Stronger activation from “{favored}”."
        metric = tooltip_estimate("Signed-rank effect", value)
        count = f"{row['n']:,} matched words"
        if row["nonzero"] != row["n"]:
            count += f" ({row['nonzero']:,} nonzero)"
    return [category, MATRIX_MODELS[model], heading, summary, metric, f"{count} · {tooltip_p(row['p'])}"]


def scatter_chart(results, categories, category_key, category_title, value_key, value_title, a, b, prime_pair):
    plotted = {model: [row for row in result["rows"] if row[value_key] is not None]
               for model, result in results.items()}
    plotted = {model: rows for model, rows in plotted.items() if rows}
    if not plotted:
        return html.P("None of the included networks has an estimate for this comparison.", role="status")
    offsets = {model: 0 if len(plotted) == 1 else .72 * (i / (len(plotted) - 1) - .5)
               for i, model in enumerate(plotted)}
    colors = {a: (ACCENT, "circle")}
    if b != a:
        colors[b] = (SECONDARY, "diamond")
    # Offsets follow the cohort order even when highlights move to the foreground.
    fig = go.Figure()
    for model in sorted(plotted, key=lambda m: m in colors):
        rows = plotted[model]
        labels = [row[category_key] for row in rows]
        values = [row[value_key] for row in rows]
        color, symbol = colors.get(model, (CONTEXT, "circle"))
        highlighted = model in colors
        error = None
        if value_key == "slope":
            error = {"type": "data", "symmetric": False, "thickness": 1.6 if highlighted else 1,
                     "width": 3 if highlighted else 1,
                     "array": [r["ci_high"] - r["slope"] if r["ci_high"] is not None else None for r in rows],
                     "arrayminus": [r["slope"] - r["ci_low"] if r["ci_low"] is not None else None for r in rows]}
        fig.add_scatter(name=MATRIX_MODELS[model], x=[categories.index(label) + offsets[model] for label in labels],
                        y=values, mode="markers", opacity=1 if highlighted else .78,
                        marker={"size": 10 if highlighted else 5, "color": color, "symbol": symbol},
                        error_y=error, customdata=[norm_tooltip(model, row, value_key, prime_pair) for row in rows],
                        hoverinfo="none")
    style_figure(fig, 375)
    # Keep touch drags available for horizontal scrolling through the categories.
    fig.update_layout(margin={"l": 50, "r": 15, "t": 16, "b": 40}, showlegend=False, dragmode=False)
    fig.update_xaxes(title_text=category_title, tickmode="array", tickvals=list(range(len(categories))),
                     ticktext=categories, range=[-.5, len(categories) - .5], showgrid=False,
                     zeroline=False, automargin=True, autotickangles=[0, 90], title_standoff=8)
    fig.update_yaxes(title_text=value_title, title_standoff=8, automargin=True, rangemode="tozero",
                     zeroline=True, zerolinewidth=2, zerolinecolor="#657b72")
    legend = [(MATRIX_MODELS[m], *style) for m, style in colors.items() if m in plotted]
    others = [m for m in plotted if m not in colors]
    if others:
        label = MATRIX_MODELS[others[0]] if len(others) == 1 else f"{len(others)} {'other ' if legend else ''}networks"
        legend.append((label, CONTEXT, "circle"))
    key = html.Ul([html.Li([html.Span(className="series-swatch", style={
        "backgroundColor": color, "width": "8px", "height": "8px",
        "borderRadius": "50%" if symbol == "circle" else 0,
        "transform": "rotate(45deg)" if symbol == "diamond" else "none",
    }), name]) for name, color, symbol in legend], className="series-legend",
        **{"aria-label": "Highlighted and included networks"})
    # At small viewport widths, give dense point groups enough horizontal room
    # to inspect; the page and the legend stay inside the viewport.
    width = 70 + len(categories) * max(22, 4 * len(plotted))
    chart = graph(fig)
    chart.className = "norm-hover-chart"
    plot = html.Div(html.Div(chart, style={"minWidth": f"{width}px"}),
                    className="table-scroll", tabIndex=0, role="region", **{"aria-label": value_title + " chart"})
    return html.Div([key, plot])


# A shared scale lets the same color represent the same estimate across networks.
MATRIX_COLORS = ["#cb8e65", "#ead2bc", "#faf9f5", "#bedbe1", "#639caf"]


def norm_chart(results, categories, category_key, category_title, value_key, value_title, a, b,
               export_title, chart_view="matrix", prime_pair=None):
    if chart_view == "scatter":
        return scatter_chart(results, categories, category_key, category_title, value_key, value_title, a, b, prime_pair)
    valence = value_key == "slope"
    values, details, cell_text = [], [], []
    for model, result in results.items():
        indexed = {row[category_key]: row for row in result["rows"]}
        cells = [indexed.get(category, {}) for category in categories]
        estimates = [cell.get(value_key) for cell in cells]
        values.append(estimates)
        cell_text.append(["" if value is None else f"{value:+.2f}" if value else "0.00" for value in estimates])
        details.append([norm_tooltip(model, cell, value_key, prime_pair) for cell in cells])

    finite = [value for row in values for value in row if value is not None]
    if not finite:
        return html.P("None of the included networks has an estimate for this comparison.", role="status")
    # The signed-rank effect is bounded by one. Slopes have no theoretical bound.
    # 1.5 covers the five published domains. Extend in half units for larger custom slopes.
    limit = max(1.5, ceil(max(map(abs, finite)) * 2) / 2) if valence else 1.0
    model_names = [MATRIX_MODELS[m] for m in results]
    fig = go.Figure(go.Heatmap(
        x=list(categories), y=model_names, z=values, zmin=-limit, zmax=limit, zmid=0,
        colorscale=MATRIX_COLORS, showscale=False, xgap=1, ygap=1, hoverongaps=False,
        text=cell_text, texttemplate="%{text}", textfont={"size": 12, "color": "#152c31"},
        customdata=details, hoverinfo="none",
    ))
    highlights = {a: ("primary", ACCENT, "●")}
    if b != a:
        highlights[b] = ("secondary", SECONDARY, "◆")
    row_headers, ticks = [], []
    for index, (model, name) in enumerate(zip(results, model_names)):
        highlight, color, symbol = highlights.get(model, ("", "", ""))
        marker = html.Span(symbol, className="norm-highlight-symbol", **{"aria-hidden": "true"}) if symbol else None
        # Keep snapshot dates together when names wrap in the pinned column.
        title, separator, variant = name.partition(" · ")
        label = html.Span([title, separator, html.Span(variant, className="norm-model-variant")]) if separator else name
        row_headers.append(html.Div([marker, label], className="norm-matrix-row",
                                    **{"data-highlight": highlight}))
        wrapped = "<br>".join(escape(part) for part in wrap(name, width=21))
        ticks.append(f"<b><span style='color:{color}'>{symbol} {wrapped}</span></b>" if symbol else wrapped)
        if highlight:
            fig.add_shape(type="rect", xref="paper", yref="paper", x0=0, x1=1,
                          y0=1 - (index + 1) / len(values), y1=1 - index / len(values),
                          line={"color": color, "width": 1.5}, fillcolor="rgba(0,0,0,0)")

    row_height, label_width = 46, 148
    cell_width = max(62, min(132, ceil(6.2 * max(map(len, categories))) + 6))
    category_lines = [wrap(cat, width=max(13, cell_width // 7)) for cat in categories]
    header_height = 30 + 15 * (max(map(len, category_lines)) - 1)
    style_figure(fig, header_height + row_height * len(values) + 2)
    # Native ticks keep the labels in Plotly image exports. The matching HTML
    # headers pin both axes during browser scrolling without moving any cells.
    fig.update_layout(margin={"t": header_height, "l": label_width, "b": 2, "r": 0, "pad": 0, "autoexpand": False},
                      plot_bgcolor="#e0e4e2", showlegend=False, dragmode=False,
                      meta={"exportTitle": escape(export_title), "metric": value_title,
                            "exportHeight": fig.layout.height + 65,
                            "filename": "word-valence" if valence else "emotion-word-sets"})
    fig.update_xaxes(type="category", side="top", categoryorder="array", categoryarray=list(categories),
                     tickmode="array", tickvals=list(categories),
                     ticktext=["<br>".join(escape(s) for s in lines) for lines in category_lines],
                     tickangle=0, fixedrange=True, showgrid=False,
                     automargin=False, range=[-.5, len(categories) - .5])
    fig.update_yaxes(type="category", categoryorder="array", categoryarray=model_names,
                     tickmode="array", tickvals=model_names, ticktext=ticks, tickfont={"size": 11},
                     fixedrange=True, showgrid=False, automargin=False, range=[len(values) - .5, -.5])

    legend = html.Div([
        html.Strong(value_title), html.Span(f"−{limit:g}"),
        html.Div([html.I(), html.Span("0")], className="norm-scale-ramp",
                 style={"--norm-gradient": "linear-gradient(to right, " + ", ".join(MATRIX_COLORS) + ")"}),
        html.Span(f"+{limit:g}"),
        *([html.Span("Missing estimate", className="norm-missing-key")]
          if any(value is None for row in values for value in row) else []),
        html.Button("Download PNG", id="norm-download", className="norm-download", n_clicks=0),
    ], className="norm-scale")
    chart = graph(fig)
    chart.className = "norm-hover-chart"
    chart.id = "norm-matrix-plot"
    chart.config = {"displayModeBar": False}
    matrix = html.Div([
        chart,
        html.Div(row_headers, className="norm-matrix-rows"),
        html.Div([html.Span(cat, title=cat) for cat in categories], className="norm-matrix-columns",
                 **{"aria-label": category_title}),
        html.Div("Network", className="norm-matrix-corner"),
    ], className="norm-matrix", style={
        "--norm-rows-width": f"{label_width}px", "--norm-header-height": f"{header_height}px",
        "--norm-row-height": f"{row_height}px", "--norm-cols": len(categories),
        "minWidth": f"{label_width + len(categories) * cell_width}px",
    })
    return html.Div([
        legend, html.Div(id="norm-download-note", role="alert"),
        html.Div(matrix, className="norm-matrix-scroll", tabIndex=0, role="region",
                 **{"aria-label": value_title + " chart"}),
    ])


def references(method):
    if method == "valence":
        credit = "Valence ratings from the NRC VAD Lexicon v1.0 by Saif M. Mohammad, National Research Council Canada."
        link = "https://saifmohammad.com/WebPages/nrc-vad.html"
    else:
        credit = "Emotion word sets from NRC EmoLex v0.92 by Saif M. Mohammad and Peter D. Turney, National Research Council Canada."
        link = "https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm"
    return html.P([credit, " ", html.A("Lexicon", href=link), " · ",
                   html.A("Study methods", href="https://arxiv.org/html/2510.24488#S2")], className="source-note")


def cohort_status(domain, models):
    if domain is None:
        return html.P("Choose a social domain.", role="status")
    if not models:
        return html.P("Select at least one network to include in this analysis.", role="status")
    if isinstance(domain, dict):
        try:
            load_lexicons()
        except NormSourceError:
            return html.P("The NRC norms are temporarily unavailable. Reopen this analysis to retry.", role="status")
    return None


def coverage(results, primes, value_key):
    notes = {}
    for model, result in results.items():
        missing = [prime for prime in primes if prime not in result["available_primes"]]
        note = result.get("reason") or ""
        if missing and not all(prime in note for prime in missing):
            note = (note + " Unavailable primes include " + ", ".join(missing) + ".").strip()
        if not any(row[value_key] is not None for row in result["rows"]):
            note = "No estimable values." + (" " + note if note else "")
        if note:
            notes.setdefault(note, []).append(MATRIX_MODELS[model])
    if not notes:
        return []
    return [html.Details([html.Summary("Network coverage notes"),
                          html.Ul([html.Li(f"{note} {' · '.join(names)}") for note, names in notes.items()])])]


def valence_view(domain, a, b, included=None, chart_view="matrix"):
    domain = resolve_domain(domain)
    models = included_models(included)
    status = cohort_status(domain, models)
    if status is not None:
        return status
    results = {m: valence_result(domain, m) for m in models}
    selected = [MATRIX_MODELS[m] for m in results if m in (a, b)]
    rows, joint_rows = [], []
    for model, result in results.items():
        for row in result["rows"]:
            rows.append([MATRIX_MODELS[model], row["prime"], f'{row["n"]:,}', number(row["slope"]),
                         number(row["ci_low"]), number(row["ci_high"]), format_p(row["p"])])
        joint = result.get("joint")
        if joint and joint.get("statistic") is not None:
            joint_rows.append([MATRIX_MODELS[model], f'{joint["n"]:,}', number(joint["statistic"]),
                               joint["df"], format_p(joint["p"]), number(joint["partial_r2"])])
    scope = "Figures 10–11 method" if domain in ("religion", "ethnicity", "orientation") else "Exploratory application"
    guide = ("Whiskers show 95% confidence intervals. Hover or tap a point for its network and slope."
             if chart_view == "scatter" else
             "Hover or tap a cell for its 95% confidence interval, word count and p-value.")
    return html.Section([
        html.H2("Activation and word valence"),
        html.P(f"{scope} · {label_for(domain)}. A positive slope means words reached more strongly by that prime "
               f"tend to have higher human pleasantness ratings. {guide}"),
        *coverage(results, primes_for(domain), "slope"),
        norm_chart(results, primes_for(domain), "prime", "Prime", "slope", "Valence slope", a, b,
                   f"{label_for(domain)} · Activation and word valence", chart_view),
        html.H3("Do slopes differ across primes?"),
        html.P("The joint Wald test compares the prime coefficients in one model of word valence. Partial R² measures the additional variation explained by allowing these coefficients to differ."),
        result_table(["Network", "Rated words", "Wald χ²", "df", "p", "Partial R²"], joint_rows,
                     selected, label="Joint valence tests") if joint_rows else html.P("The joint test could not be estimated."),
        html.Details([html.Summary("Per-prime statistics"), result_table(
            ["Network", "Prime", "Rated words", "Slope", "95% lower", "95% upper", "p"], rows,
            selected, label="Valence coefficients and intervals")]),
        html.P("The fit uses every graph word with a valence rating. Activation is L2-normalized by column "
               "and then by row over the domain’s available primes.", className="source-note"),
        activation_sources(domain, list(results)),
        references("valence"),
    ], className="panel")


def emotion_view(domain, a, b, left, right, included=None, chart_view="matrix"):
    domain = resolve_domain(domain)
    models = included_models(included)
    status = cohort_status(domain, models)
    if status is not None:
        return status
    if left == right or any(p not in primes_for(domain) for p in (left, right)):
        return html.P("Select two different primes from this domain.", role="status")
    results = {m: emotion_result(domain, m, left, right) for m in models}
    selected = [MATRIX_MODELS[m] for m in results if m in (a, b)]
    rows = []
    for model, result in results.items():
        for row in result["rows"]:
            rows.append([MATRIX_MODELS[model], row["emotion"], row["n"], row["nonzero"],
                         number(row["effect"]), format_p(row["p"])])
    scope = "Figure 12 method" if domain == "politics" else "Exploratory application"
    guide = ("Each point is one network’s effect for an emotion. Hover or tap it for the word counts and p-value."
             if chart_view == "scatter" else "Hover or tap a cell for the word counts and p-value.")
    return html.Section([
        html.H2("Activation by emotion word set"),
        html.P(f"{scope} · {label_for(domain)}. Positive effects mean words in a set received stronger activation from {left}. "
               f"Negative effects favor {right}. {guide}"),
        *coverage(results, (left, right), "effect"),
        norm_chart(results, EMOTIONS, "emotion", "Emotion word set", "effect", "Signed-rank effect (z/√N)", a, b,
                   f"{label_for(domain)} · Emotion word sets · {left} versus {right}", chart_view, (left, right)),
        html.Details([html.Summary("Paired test statistics"), result_table(
            ["Network", "Emotion", "Matched words", "Nonzero pairs", "Effect", "Two-sided p"], rows,
            selected, label="Emotion word set tests")]),
        html.P("Each pair is L1-normalized by column and then by row across the two primes. Each Wilcoxon "
               "test uses all graph words in its NRC emotion set. Coverage can differ across networks, "
               "and a word can belong to several emotion sets.", className="source-note"),
        activation_sources(domain, list(results)),
        references("emotion"),
    ], className="panel")


def register(app):
    app.clientside_callback(
        """async function(clicks, figure) {
            if (!clicks || !figure) return dash_clientside.no_update;
            try {
                const {data, layout} = structuredClone(figure);
                const {exportTitle, exportHeight, metric, filename} = layout.meta;
                const limit = data[0].zmax;
                layout.title = {text: exportTitle + '<br><sup>' + metric + ' · scale −' + limit +
                                ' to +' + limit + '</sup>', x: .02, xanchor: 'left', yanchor: 'top'};
                layout.margin.t += 65;
                layout.margin.l = 210;
                layout.margin.r = 16;
                layout.paper_bgcolor = 'white';
                const png = await Plotly.toImage({data, layout}, {format: 'png', scale: 2,
                    width: Math.max(860, 230 + 94 * data[0].x.length), height: exportHeight});
                const link = document.createElement('a');
                link.href = png;
                link.download = filename + '.png';
                link.click();
                return '';
            } catch (error) {
                console.error("Could not export the network matrix", error);
                return 'The chart could not be downloaded. Please try again.';
            }
        }""",
        Output("norm-download-note", "children"), Input("norm-download", "n_clicks"),
        State("norm-matrix-plot", "figure"), prevent_initial_call=True,
    )
    # Only the mounted panel requests a plot. A global domain change also
    # replaces the panel, so listening to it here would redraw a removed graph.
    app.callback(Output("valence-results", "children"), Input("valence-domain", "data"),
                 Input("social-a", "value"), Input("social-b", "value"), Input("included-models", "data"),
                 Input("valence-chart-view", "value"))(valence_view)
    app.callback(Output("emotion-results", "children"), Input("emotion-domain", "data"),
                 Input("social-a", "value"), Input("social-b", "value"),
                 Input("emotion-left", "value"), Input("emotion-right", "value"), Input("included-models", "data"),
                 Input("emotion-chart-view", "value"))(emotion_view)
