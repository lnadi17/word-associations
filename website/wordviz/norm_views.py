"""Valence regressions and emotion-word contrasts for social identity primes."""

import plotly.graph_objects as go
from dash import Input, Output, dcc, html

from .bias_norms import emotion_result, valence_result
from .experiments import ACCENT, SECONDARY, activation_sources, format_p, graph, result_table, style_figure
from .models import MATRIX_MODELS
from .social_domains import DOMAIN_LABELS, DOMAIN_PRIMES


def valence_layout(domain):
    return html.Div([dcc.Store(id="valence-domain", data=domain),
                     dcc.Loading(html.Div(id="valence-results"))])


def emotion_layout(domain):
    primes = DOMAIN_PRIMES[domain]
    fields = []
    for i, label in enumerate(("First prime", "Comparison prime")):
        control_id = ("emotion-left", "emotion-right")[i]
        fields.append(html.Div([
            html.Label(label, htmlFor=control_id),
            dcc.Dropdown(id=control_id, options=list(primes), value=primes[i], clearable=False,
                         persistence=domain, persistence_type="memory"),
        ], className="field"))
    return html.Div([dcc.Store(id="emotion-domain", data=domain), html.Div(fields, className="controls"),
                     dcc.Loading(html.Div(id="emotion-results"))])


def number(value):
    return "—" if value is None else f"{value:.3f}"


def norm_chart(results, category_key, category_title, value_key, value_title):
    categories = list(dict.fromkeys(row[category_key] for result in results.values() for row in result["rows"]))
    fig = go.Figure()
    for index, (model, result) in enumerate(results.items()):
        rows = [row for row in result["rows"] if row[value_key] is not None]
        labels = [row[category_key] for row in rows]
        values = [row[value_key] for row in rows]
        if not rows:
            continue
        offset = 0 if len(results) == 1 else (-.2, .2)[index]
        error = None
        if value_key == "slope":
            error = {"type": "data", "symmetric": False,
                     "array": [r["ci_high"] - r["slope"] if r["ci_high"] is not None else 0 for r in rows],
                     "arrayminus": [r["slope"] - r["ci_low"] if r["ci_low"] is not None else 0 for r in rows]}
        fig.add_scatter(name=MATRIX_MODELS[model], x=[categories.index(label) + offset for label in labels],
                        y=values, mode="markers",
                        marker={"size": 10, "color": (ACCENT, SECONDARY)[index], "symbol": ("circle", "diamond")[index]},
                        error_y=error, customdata=[[r[category_key], r["n"], format_p(r["p"])] for r in rows],
                        hovertemplate=("%{customdata[0]}<br>" + value_title +
                                       " %{y:.3f}<br>Words %{customdata[1]:,}<br>p %{customdata[2]}<extra>%{fullData.name}</extra>"))
    if not fig.data:
        return html.P("The selected networks have no estimable values for this comparison.", role="status")
    style_figure(fig, 375)
    fig.update_layout(margin={"l": 50, "r": 15, "t": 50, "b": 40}, showlegend=True,
                      legend={"orientation": "h", "y": 1.05, "yanchor": "bottom"})
    fig.update_xaxes(title_text=category_title, tickmode="array", tickvals=list(range(len(categories))),
                     ticktext=categories, range=[-.5, len(categories) - .5], showgrid=False,
                     zeroline=False, automargin=True, autotickangles=[0, 90], title_standoff=8)
    fig.update_yaxes(title_text=value_title, title_standoff=8, automargin=True, rangemode="tozero",
                     zeroline=True, zerolinewidth=2, zerolinecolor="#657b72")
    return graph(fig)


def references(method):
    if method == "valence":
        credit = "Valence ratings from the NRC VAD Lexicon v1.0 by Saif M. Mohammad, National Research Council Canada."
        link = "https://saifmohammad.com/WebPages/nrc-vad.html"
    else:
        credit = "Emotion word sets from NRC EmoLex v0.92 by Saif M. Mohammad and Peter D. Turney, National Research Council Canada."
        link = "https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm"
    return html.P([credit, " ", html.A("Lexicon", href=link), " · ",
                   html.A("Study methods", href="https://arxiv.org/html/2510.24488#S2")], className="source-note")


def comparison_models(a, b):
    return [model for model in (a, b) if model in MATRIX_MODELS]


def valence_view(domain, a, b):
    if a == b or a not in MATRIX_MODELS or domain not in DOMAIN_LABELS:
        return html.P("Choose a network and a different comparison network, or clear the comparison.", role="status")
    results = {m: valence_result(domain, m) for m in comparison_models(a, b)}
    rows, joint_rows, notes = [], [], []
    for model, result in results.items():
        for row in result["rows"]:
            rows.append([MATRIX_MODELS[model], row["prime"], f'{row["n"]:,}', number(row["slope"]),
                         number(row["ci_low"]), number(row["ci_high"]), format_p(row["p"])])
        joint = result.get("joint")
        if joint and joint.get("statistic") is not None:
            joint_rows.append([MATRIX_MODELS[model], f'{joint["n"]:,}', number(joint["statistic"]),
                               joint["df"], format_p(joint["p"]), number(joint["partial_r2"])])
        if result.get("reason"):
            notes.append(html.P(f'{MATRIX_MODELS[model]} · {result["reason"]}', role="status"))
    scope = "Figures 10–11 method" if domain in ("religion", "ethnicity", "orientation") else "Exploratory application"
    return html.Section([
        html.H2("Activation and word valence"),
        html.P(f"{scope} · {DOMAIN_LABELS[domain]}. A positive slope means words reached more strongly by that prime tend to have higher human pleasantness ratings. Whiskers show 95% confidence intervals."),
        *notes, norm_chart(results, "prime", "Prime", "slope", "Valence slope"),
        html.H3("Do slopes differ across primes?"),
        html.P("The joint Wald test compares the prime coefficients in one model of word valence. Partial R² measures the additional variation explained by allowing these coefficients to differ."),
        result_table(["Network", "Rated words", "Wald χ²", "df", "p", "Partial R²"], joint_rows,
                     label="Joint valence tests") if joint_rows else html.P("The joint test could not be estimated."),
        html.Details([html.Summary("Per-prime statistics"), result_table(
            ["Network", "Prime", "Rated words", "Slope", "95% lower", "95% upper", "p"], rows,
            label="Valence coefficients and intervals")]),
        html.P("Each network uses its available rated words. Activation is L2-normalized by column and then by row over the domain’s primes.", className="source-note"),
        activation_sources(domain, list(results)),
        references("valence"),
    ], className="panel")


def emotion_view(domain, a, b, left, right):
    if a == b or a not in MATRIX_MODELS or domain not in DOMAIN_LABELS:
        return html.P("Choose a network and a different comparison network, or clear the comparison.", role="status")
    if left == right or any(p not in DOMAIN_PRIMES[domain] for p in (left, right)):
        return html.P("Select two different primes from this domain.", role="status")
    results = {m: emotion_result(domain, m, left, right) for m in comparison_models(a, b)}
    rows, notes = [], []
    for model, result in results.items():
        for row in result["rows"]:
            rows.append([MATRIX_MODELS[model], row["emotion"], row["n"], row["nonzero"],
                         number(row["effect"]), format_p(row["p"])])
        if result.get("reason"):
            notes.append(html.P(f'{MATRIX_MODELS[model]} · {result["reason"]}', role="status"))
    scope = "Figure 12 method" if domain == "politics" else "Exploratory application"
    return html.Section([
        html.H2("Activation by emotion word set"),
        html.P(f"{scope} · {DOMAIN_LABELS[domain]}. Positive effects mean words in a set received stronger activation from {left}. Negative effects favor {right}."),
        *notes, norm_chart(results, "emotion", "Emotion word set", "effect", "Signed-rank effect (z/√N)"),
        html.Details([html.Summary("Paired test statistics"), result_table(
            ["Network", "Emotion", "Matched words", "Nonzero pairs", "Effect", "Two-sided p"], rows,
            label="Emotion word set tests")]),
        html.P("Each pair is L1-normalized by column and then by row across the two primes. Each Wilcoxon test compares the primes on the same words within a network. Coverage can differ across networks, and a word can belong to several emotion sets.", className="source-note"),
        activation_sources(domain, list(results)),
        references("emotion"),
    ], className="panel")


def register(app):
    # Only the mounted panel requests a plot. A global domain change also
    # replaces the panel, so listening to it here would redraw a removed graph.
    app.callback(Output("valence-results", "children"), Input("valence-domain", "data"),
                 Input("social-a", "value"), Input("social-b", "value"))(valence_view)
    app.callback(Output("emotion-results", "children"), Input("emotion-domain", "data"),
                 Input("social-a", "value"), Input("social-b", "value"),
                 Input("emotion-left", "value"), Input("emotion-right", "value"))(emotion_view)
