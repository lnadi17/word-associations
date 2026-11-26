"""Interactive comparisons of observed word-association neighborhoods."""
from __future__ import annotations

from collections import Counter
from html import escape
from math import sqrt

import plotly.graph_objects as go
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update

from .explore_data import DIRECTIONS, MODELS, StoreUnavailable, neighbors, vocabulary
from .model_options import model_options
from .models import model_sort_key

COLORS = ("#167264", "#ad632c", "#5669a0", "#984f66", "#547733", "#91689d", "#8d7129", "#347d99")
NOTES = {
    "forward": "Shares use all nonblank responses to the selected cue.",
    "backward": "Shares use all occurrences of the selected word as a response.",
    "merged": "Each pair takes the larger of its two directional counts. Shares use the sum of those merged weights.",
}
MAX_WORDS = 4


def _field(label, component):
    return html.Div([html.Label(label, htmlFor=component.id), component], className="field")


def layout():
    options = model_options(MODELS)
    return html.Div([
        html.Section([
            html.Div([
                _field("Words (up to 4)", dcc.Dropdown(id="ex-words", options=["cat"], value=["cat"],
                                                       multi=True, placeholder="Search and add words")),
                _field("Model", dcc.Dropdown(id="ex-model-a", options=options, optionHeight=56, value="humans",
                                             clearable=False)),
                _field("Compare model", dcc.Dropdown(id="ex-model-b", options=options, optionHeight=56,
                                                     placeholder="Add a model")),
            ], className="controls explorer-primary"),
            html.Div([
                _field("Connections", dcc.RadioItems(id="ex-direction",
                    options=[{"label": key.title(), "value": key} for key in DIRECTIONS],
                    value="forward", className="segmented")),
                _field("View", dcc.RadioItems(id="ex-view", options=[
                    {"label": "Compact list", "value": "compact"}, {"label": "Bar list", "value": "bar"},
                    {"label": "2D graph", "value": "graph"}], value="bar", className="segmented")),
            ], className="controls explorer-modes"),
        ], className="panel"),
        html.Section([
            html.Div([
                _field("Show", dcc.Dropdown(id="ex-membership", options=[{"label": "All connections", "value": "all"}],
                                          value="all", clearable=False, searchable=False)),
                _field("Order by", dcc.Dropdown(id="ex-sort", options=[
                    {"label": "Highest share", "value": "share"}, {"label": "Highest count", "value": "count"},
                    {"label": "Largest share difference", "value": "difference"}], value="share",
                    clearable=False, searchable=False)),
                _field("Display limit", dcc.Dropdown(id="ex-limit", options=[10, 20, 30], value=10,
                                                     clearable=False, searchable=False)),
            ], className="controls explorer-filters"),
            dcc.Loading(html.Div(id="ex-results", **{"aria-live": "polite"}), type="dot"),
        ], className="panel"),
    ], className="explorer-layout")


def profiles(words, a, b):
    queries = list(dict.fromkeys(words or []))
    return [(model, word) for word in queries for model in dict.fromkeys((a, b)) if model in MODELS]


def _display_order(pairs):
    # Keep the user's cue order and the original indices used by filters, colors and Δ.
    cues = list(dict.fromkeys(word for _, word in pairs))
    return sorted(range(len(pairs)), key=lambda i: (cues.index(pairs[i][1]), model_sort_key(MODELS[pairs[i][0]])))


def search_words(search, a, b, selected):
    selected = list(selected or [])
    query = (search or "").strip().casefold()
    if not query or len(selected) >= MAX_WORDS:
        return selected
    matches = set()
    try:
        for model in dict.fromkeys((a, b)):
            if model in MODELS:
                matches.update(w for w in vocabulary(model) if w.startswith(query))
    except StoreUnavailable:
        return selected
    return list(dict.fromkeys([*selected, *sorted(matches, key=lambda w: (len(w), w))[:45]]))


def membership_options(words, a, b):
    series = profiles(words, a, b)
    options = [{"label": "All connections", "value": "all"}]
    if len(series) > 1:
        options.append({"label": "Shared by all", "value": "shared"})
        if len(series) > 2:
            options.append({"label": "Shared by some", "value": "some"})
        for i in _display_order(series):
            model, word = series[i]
            options.append({"label": f"Only {word} · {MODELS[model]}", "value": f"only-{i}"})
    return options


def retained_option(options, previous_options, value):
    # Profile indices can move when a word or model is removed.
    label = next((o["label"] for o in previous_options or [] if o["value"] == value), None)
    return next((o["value"] for o in options if o["label"] == label), options[0]["value"])


def _rank(series, totals, membership, order):
    memberships = Counter(w for result in series for w in result.counts)
    choices = list(memberships)
    if membership == "shared":
        choices = [w for w in choices if memberships[w] == len(series)]
    elif membership == "some":
        choices = [w for w in choices if 1 < memberships[w] < len(series)]
    elif membership.startswith("only-"):
        index = int(membership.split("-")[1])
        choices = [w for w in choices if memberships[w] == 1 and w in series[index].counts]

    order_index = next((i for i in range(len(series)) if order == f"profile-{i}"), None)

    def score(word):
        counts = [s.counts.get(word, 0) for s in series]
        if order_index is not None:
            return counts[order_index] / totals[order_index] if totals[order_index] else 0
        shares = [value / total for value, total in zip(counts, totals) if total]
        if order == "count":
            return max(counts)
        if order == "difference" and len(shares) > 1:
            return max(shares) - min(shares)
        return max(shares, default=0)

    return sorted(choices, key=lambda w: (-score(w), w)), memberships


def _cell(result, total, word, ranks):
    if not total:
        return "—"
    count = result.counts.get(word, 0)
    rank = f" · #{ranks[count]}" if count else ""
    return f"{count:,} · {100 * count / total:.1f}%{rank}"


def _table(series, totals, words, display_order):
    columns = [html.Th("Word", scope="col")] + [html.Th(series[i].label, scope="col") for i in display_order]
    if len(series) == 2:
        columns.append(html.Th("Δ share", scope="col",
                               title=f"{series[1].label} minus {series[0].label}, in percentage points"))
    # Competition ranks count all neighbors, including those outside the display limit.
    ranks = []
    for s in series:
        frequencies = Counter(s.counts.values())
        next_rank = 1
        by_count = {}
        for count in sorted(frequencies, reverse=True):
            by_count[count] = next_rank
            next_rank += frequencies[count]
        ranks.append(by_count)
    rows = []
    for word in words:
        cells = [html.Th(word, scope="row")]
        cells.extend(html.Td(_cell(series[i], totals[i], word, ranks[i])) for i in display_order)
        if len(series) == 2:
            delta = "—"
            if all(totals):
                shares = [s.counts.get(word, 0) / total for s, total in zip(series, totals)]
                delta = f"{100 * (shares[1] - shares[0]):+.1f} pp"
            cells.append(html.Td(delta))
        rows.append(html.Tr(cells))
    caption = "Count · share · rank in full neighborhood"
    if len(series) == 2:
        caption += f". Δ = {series[1].label} minus {series[0].label}."
    return html.Div(html.Table([html.Caption(caption), html.Thead(html.Tr(columns)), html.Tbody(rows)]),
                    className="table-scroll explorer-table", tabIndex=0, role="region",
                    **{"aria-label": "Ranked word connections"})


def _bars(series, totals, words, display_order):
    fig = go.Figure()
    for index in display_order:
        s, total = series[index], totals[index]
        counts = [s.counts.get(w, 0) for w in words]
        share = [n / total if total else None for n in counts]
        fig.add_bar(name=escape(s.label), y=list(range(len(words))), x=share, orientation="h",
                    marker_color=COLORS[index % len(COLORS)],
                    text=[f"{value:.1%}" if value else "" for value in share], texttemplate="%{text}",
                    textposition="outside", cliponaxis=False, customdata=list(zip(map(escape, words), counts)),
                    hovertemplate="%{customdata[0]}<br>%{customdata[1]:,} occurrences · %{x:.2%}<extra>%{fullData.name}</extra>")
    # Dash's responsive mode overrides figure.height. Size its container instead
    # so a 30-word comparison keeps readable bars for every series.
    height = max(320, 84 + len(words) * (14 + len(series) * 16))
    fig.update_layout(barmode="group", bargap=0.25,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="system-ui, sans-serif", color="#263f38"), showlegend=False,
                      margin=dict(l=8, r=58, t=12, b=32),
                      xaxis=dict(tickformat=".0%", rangemode="tozero", gridcolor="#e1e7e3", title=None),
                      yaxis=dict(tickmode="array", tickvals=list(range(len(words))), ticktext=list(map(escape, words)),
                                 autorange="reversed", automargin=True, title=None))
    legend = html.Ul([html.Li([
        html.Span(className="series-swatch", style={"backgroundColor": COLORS[i % len(COLORS)]}),
        series[i].label,
    ]) for i in display_order], className="series-legend", **{"aria-label": "Chart series"})
    return html.Div([legend, html.Div(dcc.Graph(
        figure=fig, responsive=True, style={"height": f"{height}px"},
        config={"displaylogo": False, "displayModeBar": False},
    ), className="bar-scroll", tabIndex=0, role="region", **{"aria-label": "Connection share bar chart"})])


def _node_label(word):
    return word if len(word) <= 13 else word[:12] + "…"


def _graphs(series, totals, words, direction, display_order=None):
    # Equal row spacing on each side keeps neighboring labels apart, even at the 30-word limit.
    half = (len(words) + 1) // 2
    positions = {}
    for side, group in ((1, words[:half]), (-1, words[half:])):
        for row, word in enumerate(group):
            y = side * (1 - 2 * (row + .5) / len(group))
            positions[word] = (side * sqrt(1 - y * y), y)
    labels = {w: _node_label(w) for w in words}
    # Reserve space outside the axes for the longest displayed label on each side.
    margins = [min(114, 14 + 8 * max((len(labels[w]) for w in group), default=0))
               for group in (words[half:], words[:half])]
    min_height = max(300, 40 + half * 25)
    peak = max((s.counts.get(w, 0) / n for s, n in zip(series, totals) if n for w in words), default=1) or 1
    panels = []
    for panel_index, i in enumerate(range(len(series)) if display_order is None else display_order):
        s, total = series[i], totals[i]
        active = [w for w in words if w in s.counts]
        color = COLORS[i % len(COLORS)]
        fig = go.Figure()
        for w in active:
            x, y = positions[w]
            weight = s.counts[w] / total
            start, end = ((x, y), (0, 0)) if direction == "backward" else ((0, 0), (x, y))
            fig.add_annotation(x=end[0], y=end[1], ax=start[0], ay=start[1], xref="x", yref="y",
                               axref="x", ayref="y", text="", showarrow=True,
                               arrowhead=0 if direction == "merged" else 2, arrowsize=1.2,
                               arrowwidth=1 + 2 * weight / peak, standoff=11, startstandoff=10, arrowcolor="#899e96")
        fig.add_scatter(x=[positions[w][0] for w in active], y=[positions[w][1] for w in active],
                        text=[escape(labels[w]) for w in active], hovertext=list(map(escape, active)),
                        customdata=[[w, s.counts[w], s.counts[w] / total] for w in active],
                        mode="markers+text", marker=dict(color=color, size=[10+12*(s.counts[w]/total/peak)**.5 for w in active]),
                        textposition=["middle right" if positions[w][0] > 0 else "middle left" for w in active],
                        cliponaxis=False,
                        hovertemplate="%{hovertext}<br>%{customdata[1]:,} occurrences · %{customdata[2]:.2%}<extra></extra>")
        fig.add_scatter(x=[0], y=[0], text=[escape(_node_label(s.word))], hovertext=[escape(s.word)],
                        customdata=[[s.word]], mode="markers+text", cliponaxis=False,
                        marker=dict(size=21, color="#263e38", symbol="diamond"), textposition="bottom center",
                        hovertemplate="%{hovertext}<extra>Selected word</extra>")
        fig.update_layout(showlegend=False, hovermode="closest", clickmode="event",
                          margin=dict(l=margins[0], r=margins[1], t=18, b=18), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(family="system-ui, sans-serif", size=12), xaxis=dict(visible=False, range=[-1.05,1.05]),
                          yaxis=dict(visible=False, range=[-1,1]))
        panels.append(html.Div([html.H3([
            html.Span(className="series-swatch", style={"backgroundColor": color}), s.label,
        ]), dcc.Graph(id={"type": "ex-network", "index": panel_index},
            figure=fig, responsive=True, style={"height": f"clamp({min_height}px, 34vw, 430px)"},
            config={"displayModeBar": False, "displaylogo": False})], className="explorer-network"))
    return html.Div(panels, className="explorer-networks")


def render(words, a, b, direction, view, membership, order, limit):
    if not words:
        return html.P("Choose a word to explore.")
    if len(words) > MAX_WORDS:
        return html.P(f"Choose up to {MAX_WORDS} words at a time.")
    if a == b:
        return html.P("Choose a different comparison model, or clear it for a single-model view.")
    if a not in MODELS or (b and b not in MODELS) or direction not in DIRECTIONS:
        return html.P("Choose a model and connection direction.")
    pairs = profiles(words, a, b)
    display_order = _display_order(pairs)
    try:
        series = [neighbors(model, word, direction) for model, word in pairs]
    except StoreUnavailable as exc:
        return html.P(str(exc), role="alert")
    allowed = {o["value"] for o in membership_options(words, a, b)}
    membership = membership if membership in allowed else "all"
    totals = [s.total for s in series]
    ranked, memberships = _rank(series, totals, membership, order)
    visible = ranked[:limit if limit in (10, 20, 30) else 10]
    heading = {"forward": "Responses to", "backward": "Cues leading to", "merged": "Connections with"}[direction]
    summary = [html.H2(f"{heading} {', '.join(words)}"),
               html.P(f"{len(visible):,} of {len(ranked):,} connections shown. {NOTES[direction]}", className="source-note")]
    if len(series) == 2 and all(s.exists for s in series):
        left, right = (series[i] for i in display_order)
        shared = sum(count == 2 for count in memberships.values())
        counts = [(f"Only {left.label}", len(left.counts) - shared), ("Shared", shared),
                  (f"Only {right.label}", len(right.counts) - shared)]
        summary.append(html.Dl([
            html.Div([html.Dt(name), html.Dd(f"{count:,}")]) for name, count in counts
        ], className="overlap-counts", **{"aria-label": "Overlap across full neighborhoods"}))
    elif len(series) > 2 and all(s.exists for s in series):
        shared = sum(count == len(series) for count in memberships.values())
        summary.append(html.P(f"{shared:,} shared by all · {len(memberships):,} distinct connections in total", className="source-note"))
    for i in display_order:
        s, total = series[i], totals[i]
        if not s.exists:
            summary.append(html.P(f"{s.word} is absent from the {MODELS[s.model]} trial vocabulary."))
        elif not total:
            summary.append(html.P(f"{s.label} has no observed {direction} connections."))
    if membership != "all" and any(not s.exists for s in series):
        return [*summary, html.P("Overlap requires every selected word to be present. Choose All connections to see the available neighborhoods.")]
    if not visible:
        return [*summary, html.P("No connections match this selection.")]
    if view == "graph":
        summary.append(html.P("Select a node to add its word. Node size and edge width show connection share.", className="source-note"))
        plot = _graphs(series, totals, visible, direction, display_order)
    elif view == "compact":
        plot = _table(series, totals, visible, display_order)
    else:
        plot = _bars(series, totals, visible, display_order)
    return [*summary, plot]


def register(app):
    app.callback(Output("ex-words", "options"), Input("ex-words", "search_value"),
                 Input("ex-model-a", "value"), Input("ex-model-b", "value"),
                 Input("ex-words", "value"))(search_words)

    @app.callback(Output("ex-sort", "options"), Output("ex-sort", "value"),
                  Input("ex-words", "value"), Input("ex-model-a", "value"), Input("ex-model-b", "value"),
                  State("ex-sort", "options"), State("ex-sort", "value"))
    def order_options(words, a, b, previous_options, value):
        series = profiles(words, a, b)
        options = [{"label": "Highest share", "value": "share"}, {"label": "Highest count", "value": "count"}]
        if len(series) > 1:
            options.append({"label": "Largest share difference", "value": "difference"})
            for i in _display_order(series):
                model, word = series[i]
                options.append({"label": f"Share · {word} · {MODELS[model]}", "value": f"profile-{i}"})
        return options, retained_option(options, previous_options, value)

    @app.callback(Output("ex-membership", "options"), Output("ex-membership", "value"),
                  Input("ex-words", "value"), Input("ex-model-a", "value"), Input("ex-model-b", "value"),
                  State("ex-membership", "options"), State("ex-membership", "value"))
    def update_membership(words, a, b, previous_options, value):
        options = membership_options(words, a, b)
        return options, retained_option(options, previous_options, value)

    @app.callback(Output("ex-words", "value"), Input({"type": "ex-network", "index": ALL}, "clickData"),
                  State("ex-words", "value"), prevent_initial_call=True)
    def select_graph_word(clicks, selected):
        selected = selected or []
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict) or len(selected) >= MAX_WORDS:
            return no_update
        index = triggered["index"]
        click = clicks[index] if index < len(clicks) else None
        if not click or not click.get("points"):
            return no_update
        data = click["points"][0].get("customdata")
        if not data or data[0] in selected:
            return no_update
        return [*selected, data[0]]

    app.callback(Output("ex-results", "children"), Input("ex-words", "value"), Input("ex-model-a", "value"),
                 Input("ex-model-b", "value"), Input("ex-direction", "value"), Input("ex-view", "value"),
                 Input("ex-membership", "value"), Input("ex-sort", "value"), Input("ex-limit", "value"))(render)
