"""Valence regressions and emotion-word contrasts for social identity primes."""

import plotly.graph_objects as go
from dash import Input, Output, dcc, html

from .bias_norms import EMOTIONS, emotion_result, valence_result
from .experiments import ACCENT, SECONDARY, activation_sources, format_p, graph, result_table, style_figure
from .model_inclusion import included_models
from .models import MATRIX_MODELS
from .norm_lexicons import NormSourceError, load_lexicons
from .social_domains import key_for, label_for, primes_for, resolve_domain


def valence_layout(domain):
    return html.Div([dcc.Store(id="valence-domain", data=domain),
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
    return html.Div([dcc.Store(id="emotion-domain", data=domain), html.Div(fields, className="controls"),
                     dcc.Loading(html.Div(id="emotion-results"))])


def number(value):
    return "—" if value is None else f"{value:.3f}"


CONTEXT = "#849491"


def norm_chart(results, categories, category_key, category_title, value_key, value_title, a, b):
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
    # Highlighted points are drawn last; changing the dropdowns never moves a point.
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
                        error_y=error, customdata=[[r[category_key], r["n"], format_p(r["p"])] for r in rows],
                        hovertemplate=("%{customdata[0]}<br>" + value_title +
                                       " %{y:.3f}<br>Words %{customdata[1]:,}<br>p %{customdata[2]}<extra>%{fullData.name}</extra>"))
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
    plot = html.Div(html.Div(graph(fig), style={"minWidth": f"{width}px"}),
                    className="table-scroll", tabIndex=0, role="region", **{"aria-label": value_title + " chart"})
    return html.Div([key, plot])


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


def valence_view(domain, a, b, included=None):
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
    return html.Section([
        html.H2("Activation and word valence"),
        html.P(f"{scope} · {label_for(domain)}. A positive slope means words reached more strongly by that prime tend to have higher human pleasantness ratings. Whiskers show 95% confidence intervals."),
        *coverage(results, primes_for(domain), "slope"),
        norm_chart(results, primes_for(domain), "prime", "Prime", "slope", "Valence slope", a, b),
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


def emotion_view(domain, a, b, left, right, included=None):
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
    return html.Section([
        html.H2("Activation by emotion word set"),
        html.P(f"{scope} · {label_for(domain)}. Positive effects mean words in a set received stronger activation from {left}. Negative effects favor {right}."),
        *coverage(results, (left, right), "effect"),
        norm_chart(results, EMOTIONS, "emotion", "Emotion word set", "effect", "Signed-rank effect (z/√N)", a, b),
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
    # Only the mounted panel requests a plot. A global domain change also
    # replaces the panel, so listening to it here would redraw a removed graph.
    app.callback(Output("valence-results", "children"), Input("valence-domain", "data"),
                 Input("social-a", "value"), Input("social-b", "value"), Input("included-models", "data"))(valence_view)
    app.callback(Output("emotion-results", "children"), Input("emotion-domain", "data"),
                 Input("social-a", "value"), Input("social-b", "value"),
                 Input("emotion-left", "value"), Input("emotion-right", "value"), Input("included-models", "data"))(emotion_view)
