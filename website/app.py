"""Flask entry point for Vercel and the local Dash development server."""

import os
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

# Plotly reads NumPy directly from sys.modules when validating chart arrays.
# Finish importing it before concurrent Flask requests can see a partial module.
import numpy  # noqa: F401
from dash import Dash, Input, Output, State, dcc, html, no_update
from flask import Flask, Response, request

from wordviz import custom_domains, experiments, explore, model_details, model_inclusion, norm_views, paths
from wordviz.social_domains import DOMAIN_LABELS, key_for


ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
dashboard = Dash(
    __name__,
    server=app,
    assets_folder=str(ROOT / "public" / "assets"),
    assets_url_path="assets",
    include_assets_files=False,
    external_stylesheets=["/assets/site.css"],
    external_scripts=["/vendor/plotly.min.js", "/assets/dropdown-labels.js", "/assets/norm-tooltip.js"],
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


@lru_cache(maxsize=1)
def mounted_graph_bundle():
    """Backport a missing mounted-element check to the pinned Dash 3.3.0 graph."""
    source = files("dash.dcc").joinpath("async-graph.js").read_text(encoding="utf-8")
    start = "d=this.gd.current;"
    transform = "a=e._dashprivate_transformFigure(a,d)"
    if source.count(start + transform) != 1:
        raise RuntimeError("Review the graph unmount guard after updating Dash")
    # plot() also runs from a promise queue. After navigation, its ref can
    # already be null before the first transform or classList access.
    return source.replace(start + transform, start + "if (!d) return Promise.resolve();" + transform)


@app.get("/_dash-component-suites/dash/dcc/async-graph.js")
def graph_component():
    response = Response(mounted_graph_bundle(), mimetype="text/javascript")
    response.add_etag()
    return response.make_conditional(request)


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
        "Compare how social identity primes activate words in human and language-model networks.",
    ),
    "/models": (
        "Model details",
        "Model and collection details",
        "Open a dataset to see its recorded version, collection details and sources.",
    ),
}


def page_content(path):
    if path == "/":
        return explore.layout()
    if path == "/lexical-decision":
        return experiments.ldt_layout()
    if path == "/models":
        return model_details.layout()
    return html.Div([
        html.Fieldset([
            html.Legend("Analysis method", className="visually-hidden"),
            dcc.RadioItems(id="bias-view", options=[
                {"label": "Priming effects", "value": "priming"},
                {"label": "Heatmaps", "value": "heatmaps"},
                {"label": "Word valence", "value": "valence"},
                {"label": "Emotion word sets", "value": "emotion"},
                {"label": "Association paths", "value": "paths"},
            ], value="priming", persistence=True, persistence_type="memory",
                className="segmented method-tabs"),
        ], className="method-selector"),
        html.Div([
            experiments.field("Social domain", dcc.Dropdown(id="social-domain",
                options=[{"label": name, "value": key} for key, name in DOMAIN_LABELS.items()],
                value="gender", clearable=False, searchable=False)),
            html.Div(experiments.model_fields("social"), id="social-models", className="model-selectors"),
        ], className="controls social-toolbar"),
        model_inclusion.layout(),
        custom_domains.layout(),
        dcc.Store(id="session-targets"),
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


for view in (explore, experiments, paths, norm_views, custom_domains, model_inclusion):
    view.register(dashboard)


dashboard.clientside_callback(
    """function(method) {
        return ["priming", "valence", "emotion"].includes(method)
            ? ["Highlight network", "Second highlight"] : ["Network", "Compare with"];
    }""",
    Output("social-a-label", "children"), Output("social-b-label", "children"), Input("bias-view", "value"),
)


@dashboard.callback(Output("bias-content", "children"), Output("social-models", "hidden"),
                    Output("bias-content", "hidden"),
                    Input("bias-view", "value"), Input("social-domain", "value"),
                    Input("custom-domains", "data"),
                    State("social-a", "value"), State("social-b", "value"), State("session-targets", "data"))
def open_bias_view(method, domain, saved_domains, first_model, second_model, session_targets):
    layouts = {"priming": experiments.priming_layout, "heatmaps": experiments.heatmaps_layout,
               "valence": norm_views.valence_layout, "emotion": norm_views.emotion_layout,
               "paths": paths.layout}
    if domain == custom_domains.NEW_DOMAIN:
        # An earlier analysis may still be writing to its results panel.
        return no_update, True, True
    if not isinstance(domain, str):
        return html.P("Select a social domain.", role="status"), False, False
    if domain not in DOMAIN_LABELS:
        domain = custom_domains.get_domain(saved_domains, domain)
    if domain is None or method not in layouts:
        return html.P("Select a method and a social domain.", role="status"), False, False
    if method == "paths":
        view = paths.layout(domain, (first_model or "Humans").lower(), second_model.lower() if second_model else None)
    elif method in ("priming", "heatmaps"):
        view = layouts[method](domain, session_targets)
    else:
        view = layouts[method](domain)
    return html.Div(view, key=f"{method}-{key_for(domain)}"), method == "paths", False


if __name__ == "__main__":
    dashboard.run(host="127.0.0.1", port=int(os.environ.get("PORT", "8050")), debug=False)
