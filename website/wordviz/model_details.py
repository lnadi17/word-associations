"""Collection records for the datasets used by the explorer and analyses."""

import json
from pathlib import Path

from dash import html

from .model_provenance import MODEL_PROVENANCE, REPO
from .models import GRAPH_MODELS, MATRIX_MODELS

RELEASE = json.loads((Path(__file__).with_name("resources") / "network_release.json").read_text())


def _record(key, record):
    human = key == "humans"
    identifier_label = record.get("identifier_label") or ("Data source" if human else "Model identifier")
    identifier = record.get("identifier") or ("Human participants" if human else "Unrecorded")
    if key in {name.lower() for name in MATRIX_MODELS}:
        usage = "Explore, Lexical decision and Implicit bias"
    elif key in GRAPH_MODELS:
        usage = "Explore and Association paths"
    else:
        usage = record.get("availability", "Collected responses")

    fields = [("Availability", usage)]
    if key in RELEASE["networks"]:
        url = f"{REPO}/releases/download/{RELEASE['tag']}/{key}.zip"
        size_mb = RELEASE["networks"][key]["bytes"] / 1_000_000
        fields.append(("Research data", html.A(
            f"Cleaned trials, graph and activation matrices (ZIP, {size_mb:.1f} MB)", href=url,
        )))
    if not human or record.get("revision"):
        fields.append(("Version or revision", record.get("revision") or "Unrecorded"))
    if record.get("collection"):
        fields.append(("Collection", record["collection"]))
    fields.extend(record.get("details", ()))
    if record.get("notes"):
        fields.append(("Notes", record["notes"]))
    sources = record["sources"]
    if sources:
        fields.append(("Sources", html.Ul(
            [html.Li(html.A(label, href=url)) for label, url in sources],
            className="model-sources",
        )))

    return html.Details([
        html.Summary([
            html.Span(record["label"], className="model-name"),
            html.Span([html.Span(identifier_label, className="model-field-label"),
                       html.Code(identifier)], className="model-identifier"),
        ]),
        html.Dl([html.Div([html.Dt(label), html.Dd(value)]) for label, value in fields],
                className="model-fields"),
    ], id=f"model-{key}", className="model-record")


def layout():
    available, additional = [], []
    for key, record in MODEL_PROVENANCE.items():
        group = available if key in GRAPH_MODELS else additional
        group.append(_record(key, record))
    return html.Div([
        html.Section([html.H2(heading), html.Div(records, className="model-list")])
        for heading, records in (("Analysis datasets", available), ("Additional collections", additional))
        if records
    ], className="model-catalog")
