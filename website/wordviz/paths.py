"""Dash section for the association paths in bias Figures 8 and 9."""

from dash import Input, Output, State, dcc, html, set_props
import plotly.graph_objects as go

from .path_data import MODEL_NAMES, load_graph, stream
from .model_options import model_dropdown, model_options
from .models import ordered_model_keys
from .social_domains import key_for, primes_for, resolve_domain


DEFAULT_PRIMES = ["feminine", "masculine"]
DEFAULT_TARGETS = ["compassionate", "forceful"]
VALENCE_COLORS = [[0, "#ad3f46"], [0.5, "#ecebe4"], [1, "#3266a4"]]


def _field(label, component, persistence=True):
    component.persistence = persistence
    component.persistence_type = "memory"
    return html.Div([html.Label(label, htmlFor=component.id), component], className="field")


def layout(domain="gender", primary_model="humans", comparison_model=None):
    options = model_options(MODEL_NAMES)
    starts = DEFAULT_PRIMES if domain == "gender" else primes_for(domain)
    ends = DEFAULT_TARGETS if domain == "gender" else ["community"]
    extra_fields = []
    persistence = key_for(domain)
    note = ""
    if isinstance(domain, dict):
        names = list(domain["lists"])
        starts, ends = starts[:2], []
        if names:
            ends = domain["lists"][names[0]][:2]
            extra_fields = [_field("Saved target list", dcc.Dropdown(id="path-list", options=names,
                                   value=names[0], clearable=False), persistence)]
            note = " Each list starts with up to two targets. Select other words to trace additional routes."
        else:
            note = " Choose a target word to trace a route from your primes."
    elif domain != "gender":
        note = " The example target is community. Choose any graph node to explore another target."
    legend = []
    for label, color in [("Negative", VALENCE_COLORS[0][1]), ("Neutral", VALENCE_COLORS[1][1]),
                         ("Positive", VALENCE_COLORS[2][1]), ("Unrated", "#aeb3b6")]:
        unrated = label == "Unrated"
        swatch = html.Span(style={
            "display": "inline-block", "width": "10px", "height": "10px",
            "backgroundColor": color, "border": "1px solid #59675d",
            "borderRadius": "0" if unrated else "50%", "transform": "rotate(45deg)" if unrated else None,
        }, **{"aria-hidden": "true"})
        legend.append(html.Li([swatch, label], style={"display": "flex", "alignItems": "center", "gap": "6px"}))
    return html.Section([
        dcc.Store(id="path-domain", data=domain),
        html.H2("Association paths", id="association-paths"),
        html.P(("Figures 8–9 · " if domain == "gender" else "")
               + "Trace the shortest routes between words in the filtered association graphs. "
               "Color shows human-rated valence from NRC VAD v2.1."
               + note),
        html.Div([
            _field("Path model", model_dropdown("path-a", options, value=primary_model, clearable=False)),
            _field("Compare path model", model_dropdown("path-b", options, value=comparison_model, placeholder="Single graph")),
            *extra_fields,
            _field("Starting words", dcc.Dropdown(
                id="path-start", value=list(starts),
                multi=True, placeholder="Search graph nodes"), persistence),
            _field("Target words", dcc.Dropdown(
                id="path-end", value=list(ends),
                multi=True, placeholder="Search graph nodes"), persistence),
        ], className="controls"),
        html.Ul(legend, className="source-note", **{"aria-label": "Word valence legend"},
                style={"display": "flex", "flexWrap": "wrap", "gap": "6px 18px", "padding": 0}),
        dcc.Loading(html.Div(id="path-results"), type="dot"),
    ], className="panel")


def choices(search, a, b, selected, suggested=()):
    selected = list(selected or [])
    query = (search or "").strip().casefold()
    if not query:
        return list(dict.fromkeys(selected + list(suggested)))
    graphs = [load_graph(m) for m in dict.fromkeys([a, b]) if m in MODEL_NAMES]
    matches = {w for g in graphs for w in g["words"] if w.startswith(query)}
    return list(dict.fromkeys(selected + sorted(matches, key=lambda w: (len(w), w))[:40]))


def _path_figure(result, max_hops):
    layers = result["layers"]
    peak = max(map(len, layers))
    positions = {
        w: (depth, (len(layer) - 1) / 2 - i)
        for depth, layer in enumerate(layers) for i, w in enumerate(layer)
    }
    edge_x, edge_y = [], []
    for src, tgt in result["visible_edges"]:
        edge_x.extend([positions[src][0], positions[tgt][0], None])
        edge_y.extend([positions[src][1], positions[tgt][1], None])
    fig = go.Figure(go.Scatter(
        x=edge_x, y=edge_y, mode="lines", line={"color": "#bcc6c0", "width": 1},
        hoverinfo="skip", showlegend=False,
    ))
    for rated in (True, False):
        words = [w for w in positions if (result["valence"][w] is not None) == rated]
        if not words:
            continue
        values = [result["valence"][w] for w in words]
        marker = dict(size=12, line={"width": 1, "color": "#59675d"})
        if rated:
            marker.update(color=values, colorscale=VALENCE_COLORS, cmin=-1, cmax=1)
        else:
            marker.update(color="#aeb3b6", symbol="diamond")
        fig.add_trace(go.Scatter(
            x=[positions[w][0] for w in words], y=[positions[w][1] for w in words],
            text=words, mode="markers+text", textposition="top center", textfont={"size": 11},
            marker=marker, showlegend=False, cliponaxis=False,
            customdata=["unrated" if v is None else f"{v:+.3f}" for v in values],
            hovertemplate="%{text}<br>Hop %{x}<br>Valence %{customdata}<extra></extra>",
        ))
    fig.update_layout(
        height=max(190, 30 * peak + 85), margin={"l": 42, "r": 42, "t": 26, "b": 28},
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "system-ui, sans-serif", "color": "#273b33"},
        xaxis=dict(range=[-.2, max(max_hops, 1) + .2], dtick=1, showgrid=False,
                   zeroline=False, title=None, tickfont=dict(color="#67766b")),
        yaxis=dict(visible=False, range=[-max(1, peak / 2), max(1, peak / 2)]),
    )
    return fig


def _model_panel(model, result, max_hops):
    content = [html.H4(MODEL_NAMES[model])]
    if "missing" in result:
        content.append(html.P("Missing graph nodes: " + ", ".join(result["missing"])))
    elif result.get("unreachable"):
        content.append(html.P("These nodes are in disconnected components."))
    else:
        metrics = (
            f'{result["routes"]:,} shortest {"route" if result["routes"] == 1 else "routes"}'
            f' · {result["hops"]} {"hop" if result["hops"] == 1 else "hops"}'
            f' · {result["node_count"]:,} nodes'
        )
        content.append(html.P(metrics, className="source-note"))
        if len(result["visible_edges"]) < result["edge_count"]:
            content.append(html.P(
                f'Showing complete routes covering {len(result["valence"]):,} of '
                f'{result["node_count"]:,} nodes and {len(result["visible_edges"]):,} of '
                f'{result["edge_count"]:,} links.', className="source-note",
            ))
        figure = _path_figure(result, max_hops)
        content.append(dcc.Graph(
            figure=figure, responsive=True, style={"height": f"{figure.layout.height}px"},
            config={"displaylogo": False, "displayModeBar": "hover", "scrollZoom": False},
        ))
        content.append(html.Small("Filtered graph · " + result["source"], className="source-note"))
    return html.Div(content, className="path-model", style={"minWidth": 0})


def render(a, b, starts, ends):
    if a == b and a in MODEL_NAMES:
        return html.P("Choose a different comparison graph, or clear Compare path model.", role="status")
    if not starts or not ends:
        return html.P("Select at least one starting word and one target word.")
    if len(starts) * len(ends) > 12:
        return html.P("Choose up to 12 starting-word and target-word combinations per view.")
    models = ordered_model_keys(m for m in (a, b) if m in MODEL_NAMES)
    if not models:
        return html.P("Select a graph.")
    views = []
    for prime in starts:
        for target in ends:
            results = {model: stream(model, prime, target) for model in models}
            max_hops = max((r.get("hops", 0) for r in results.values()), default=0)
            panels = [_model_panel(m, r, max_hops) for m, r in results.items()]
            views.append(html.Div([
                html.H3(f"{prime} → {target}"),
                html.Div(panels, className="path-comparison", style={
                    "display": "grid", "gap": "1rem",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(min(100%, 330px), 1fr))",
                }),
            ], className="path-pair"))
    return views


def register(app):
    @app.callback(Output("path-start", "options"), Input("path-start", "search_value"), Input("path-a", "value"),
                  Input("path-b", "value"), State("path-start", "value"), State("path-domain", "data"))
    def start_choices(query, a, b, selected, domain):
        domain = resolve_domain(domain)
        return choices(query, a, b, selected, domain["primes"] if isinstance(domain, dict) else ())

    @app.callback(Output("path-end", "options"), Input("path-end", "search_value"), Input("path-a", "value"),
                  Input("path-b", "value"), Input("path-list", "value", allow_optional=True),
                  State("path-end", "value"), State("path-domain", "data"))
    def end_choices(query, a, b, name, selected, domain):
        domain = resolve_domain(domain)
        words = domain["lists"].get(name, ()) if isinstance(domain, dict) and isinstance(name, str) else ()
        return choices(query, a, b, selected, words)

    @app.callback(Input("path-list", "value"), State("path-domain", "data"), prevent_initial_call=True)
    def switch_list(name, domain):
        domain = resolve_domain(domain)
        if isinstance(domain, dict) and isinstance(name, str) and name in domain["lists"]:
            set_props("path-end", {"value": domain["lists"][name][:2]})

    app.callback(Output("path-results", "children"), Input("path-a", "value"),
                 Input("path-b", "value"), Input("path-start", "value"),
                 Input("path-end", "value"))(render)
