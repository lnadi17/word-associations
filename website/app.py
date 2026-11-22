"""Flask entry point for Vercel and the local Dash development server."""

import os
from pathlib import Path

from dash import Dash, Input, Output, dcc, html
from flask import Flask

from wordviz import experiments, explore, paths


ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
dashboard = Dash(
    __name__,
    server=app,
    assets_folder=str(ROOT / "public" / "assets"),
    assets_url_path="assets",
    include_assets_files=False,
    external_stylesheets=["/assets/site.css"],
    external_scripts=["/vendor/plotly.min.js", "/assets/dropdown-labels.js"],
    suppress_callback_exceptions=True,
    title="Word Associations",
    meta_tags=[{
        "name": "description",
        "content": "Explore human and language-model word associations, compare networks, and examine lexical-decision and priming experiments.",
    }],
    update_title=None,
)
dashboard.index_string = dashboard.index_string.replace(
    "{%favicon%}", '<link rel="icon" type="image/svg+xml" href="/assets/word-associations.svg">'
)


@app.get("/vendor/plotly.min.js")
def plotly_js():
    # Vercel serves this from public/vendor on its CDN; local runs use the wheel.
    from flask import send_file
    from plotly import __file__ as plotly_file

    return send_file(Path(plotly_file).parent / "package_data" / "plotly.min.js")


PAGES = {
    "/": (
        "Explore",
        "Association explorer",
        "Look up a word to see what it evokes, or trace which cues lead back to it.",
    ),
    "/lexical-decision": (
        "Lexical decision",
        "Lexical decision",
        "Compare target activation after related and unrelated primes against human reaction times.",
    ),
    "/implicit-bias": (
        "Implicit bias",
        "Implicit associations",
        "Examine priming effects, compare target activation, and trace the paths between words.",
    ),
}


def page_content(path):
    if path == "/":
        return explore.layout()
    if path == "/lexical-decision":
        return experiments.ldt_layout()
    return html.Div([
        dcc.RadioItems(
            id="bias-view",
            options=[
                {"label": "Priming effects and heatmaps", "value": "activation"},
                {"label": "Association paths", "value": "paths"},
            ],
            value="activation",
            className="segmented bias-view",
        ),
        html.Div(id="bias-content"),
    ])


def navigation(active):
    return html.Nav([
        dcc.Link(label, href=path, className="nav-link active" if active == path else "nav-link")
        for path, (label, _, _) in PAGES.items()
    ], **{"aria-label": "Main navigation"})


dashboard.layout = html.Div([
    dcc.Location(id="location", refresh=False),
    html.Header(html.Div([
        dcc.Link([
            html.Span([html.I(), html.I(), html.I()], className="brand-symbol", **{"aria-hidden": "true"}),
            html.Span("Word Associations"),
        ], href="/", className="wordmark"),
        html.Div(id="nav"),
    ], className="masthead-inner"), className="masthead"),
    html.Main(id="workspace", className="workspace"),
])


@dashboard.callback(Output("workspace", "children"), Output("nav", "children"), Input("location", "pathname"))
def open_page(pathname):
    path = (pathname or "/").rstrip("/") or "/"
    if path not in PAGES:
        return html.Div([html.H1("Page not found"), dcc.Link("Return to the explorer", href="/")]), navigation(None)
    _, title, description = PAGES[path]
    return [
        html.Div([html.H1(title), html.P(description)], className="page-heading"),
        page_content(path),
    ], navigation(path)


for view in (explore, experiments, paths):
    view.register(dashboard)


@dashboard.callback(Output("bias-content", "children"), Input("bias-view", "value"))
def open_bias_view(selected):
    return paths.layout() if selected == "paths" else experiments.bias_layout()


if __name__ == "__main__":
    dashboard.run(host="127.0.0.1", port=int(os.environ.get("PORT", "8050")), debug=False)
